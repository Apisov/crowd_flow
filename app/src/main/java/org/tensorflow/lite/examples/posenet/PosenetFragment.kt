/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.posenet

import android.Manifest
import android.animation.Animator
import android.animation.ValueAnimator
import android.app.Activity
import android.app.AlertDialog
import android.app.Dialog
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PorterDuff
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.media.Image
import android.media.ImageReader
import android.media.ImageReader.OnImageAvailableListener
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.support.v4.app.ActivityCompat
import android.support.v4.app.DialogFragment
import android.support.v4.app.Fragment
import android.support.v4.content.ContextCompat
import android.util.DisplayMetrics
import android.util.Log
import android.util.Size
import android.util.SparseIntArray
import android.view.LayoutInflater
import android.view.Surface
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.view.ViewGroup
import android.widget.SeekBar
import android.widget.Toast
import kotlinx.android.synthetic.main.activity_posenet.*
import org.puredata.android.io.AudioParameters
import org.puredata.android.io.PdAudio
import org.puredata.core.PdBase
import org.puredata.core.utils.IoUtils
import org.tensorflow.lite.examples.posenet.lib.BodyPart
import org.tensorflow.lite.examples.posenet.lib.KeyPoint
import org.tensorflow.lite.examples.posenet.lib.Person
import org.tensorflow.lite.examples.posenet.lib.Posenet
import java.io.File
import java.io.IOException
import java.util.concurrent.Semaphore
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

class PosenetFragment : Fragment(), ActivityCompat.OnRequestPermissionsResultCallback {

    /** List of body joints that should be connected.    */
    private val bodyJoints = listOf(
        Pair(BodyPart.LEFT_WRIST, BodyPart.LEFT_ELBOW),
        Pair(BodyPart.LEFT_ELBOW, BodyPart.LEFT_SHOULDER),
        Pair(BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER),
        Pair(BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_ELBOW),
        Pair(BodyPart.RIGHT_ELBOW, BodyPart.RIGHT_WRIST)
    )

    private val trackingBodyParts = listOf(
        BodyPart.LEFT_WRIST,
        BodyPart.LEFT_ELBOW,
        BodyPart.LEFT_SHOULDER,
        BodyPart.RIGHT_SHOULDER,
        BodyPart.RIGHT_ELBOW,
        BodyPart.RIGHT_WRIST,
        BodyPart.LEFT_EYE,
        BodyPart.RIGHT_EYE
    )

    /** Threshold for confidence score. */
    private val minConfidence = 0.5

    /** Radius of circle used to draw keypoints.  */
    private val circleRadius = 8.0f

    /** Paint class holds the style and color information to draw geometries,text and bitmaps. */
    private var paint = Paint()

    /** A shape for extracting frame data.   */
    private val PREVIEW_WIDTH = 640
    private val PREVIEW_HEIGHT = 480

    /** An object for the Posenet library.    */
    private lateinit var posenet: Posenet

    /** ID of the current [CameraDevice].   */
    private var cameraId: String? = null

    /** A [SurfaceView] for camera preview.   */
    private var surfaceView: SurfaceView? = null

    /** A [CameraCaptureSession] for camera preview.   */
    private var captureSession: CameraCaptureSession? = null

    /** A reference to the opened [CameraDevice].    */
    private var cameraDevice: CameraDevice? = null

    /** The [android.util.Size] of camera preview.  */
    private var previewSize: Size? = null

    /** The [android.util.Size.getWidth] of camera preview. */
    private var previewWidth = 0

    /** The [android.util.Size.getHeight] of camera preview.  */
    private var previewHeight = 0

    /** A counter to keep count of total frames.  */
    private var frameCounter = 0

    /** An IntArray to save image data in ARGB8888 format  */
    private lateinit var rgbBytes: IntArray

    /** A ByteArray to save image data in YUV format  */
    private var yuvBytes = arrayOfNulls<ByteArray>(3)

    /** An additional thread for running tasks that shouldn't block the UI.   */
    private var backgroundThread: HandlerThread? = null

    /** A [Handler] for running tasks in the background.    */
    private var backgroundHandler: Handler? = null

    /** An [ImageReader] that handles preview frame capture.   */
    private var imageReader: ImageReader? = null

    /** [CaptureRequest.Builder] for the camera preview   */
    private var previewRequestBuilder: CaptureRequest.Builder? = null

    /** [CaptureRequest] generated by [.previewRequestBuilder   */
    private var previewRequest: CaptureRequest? = null

    /** A [Semaphore] to prevent the app from exiting before closing the camera.    */
    private val cameraOpenCloseLock = Semaphore(1)

    /** Whether the current camera device supports Flash or not.    */
    private var flashSupported = false

    /** Orientation of the camera sensor.   */
    private var sensorOrientation: Int? = null

    /** Abstract interface to someone holding a display surface.    */
    private var surfaceHolder: SurfaceHolder? = null

    private var playing = false
    private var startPlayingTime = 0L
    private var lSideConfidence = 0
    private var rSideConfidence = 0
    private var currentSide = NONE

    private var stopConfidence = 0

    private val volumeAnimator: ValueAnimator = ValueAnimator.ofFloat().apply {
        duration = 50
        addListener(object : Animator.AnimatorListener {
            override fun onAnimationRepeat(animation: Animator?) {
                Log.i(TAG, "onAnimationRepeat")
            }

            override fun onAnimationEnd(animation: Animator?) {
                Log.i(TAG, "onAnimationEnd")
            }

            override fun onAnimationCancel(animation: Animator?) {
                Log.i(TAG, "onAnimationCancel")
            }

            override fun onAnimationStart(animation: Animator?) {
                Log.i(TAG, "onAnimationStart")
            }
        })
        addUpdateListener {
            val tmpVolume = it.animatedValue as Float
            if (tmpVolume > 0) {
                volume = it.animatedValue as Float
                PdBase.sendFloat("master_volume", volume)
            }
        }
    }
    var volume: Float = 30F

    /** [CameraDevice.StateCallback] is called when [CameraDevice] changes its state.   */
    private val stateCallback = object : CameraDevice.StateCallback() {

        override fun onOpened(cameraDevice: CameraDevice) {
            cameraOpenCloseLock.release()
            this@PosenetFragment.cameraDevice = cameraDevice
            createCameraPreviewSession()
        }

        override fun onDisconnected(cameraDevice: CameraDevice) {
            cameraOpenCloseLock.release()
            cameraDevice.close()
            this@PosenetFragment.cameraDevice = null
        }

        override fun onError(cameraDevice: CameraDevice, error: Int) {
            onDisconnected(cameraDevice)
            this@PosenetFragment.activity?.finish()
        }
    }

    /**
     * A [CameraCaptureSession.CaptureCallback] that handles events related to JPEG capture.
     */
    private val captureCallback = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureProgressed(
            session: CameraCaptureSession,
            request: CaptureRequest,
            partialResult: CaptureResult
        ) {
        }

        override fun onCaptureCompleted(
            session: CameraCaptureSession,
            request: CaptureRequest,
            result: TotalCaptureResult
        ) {
        }
    }

    /**
     * Shows a [Toast] on the UI thread.
     *
     * @param text The message to show
     */
    private fun showToast(text: String) {
        val activity = activity
        activity?.runOnUiThread { Toast.makeText(activity, text, Toast.LENGTH_SHORT).show() }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? = inflater.inflate(R.layout.activity_posenet, container, false)

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        surfaceView = view.findViewById(R.id.surfaceView)
        surfaceHolder = surfaceView!!.holder
        val listener = object : SimpleSeekBarListener() {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                when (seekBar) {
                    pitch -> PdBase.sendFloat("pitch_slider", progress.toFloat())
                    dark -> PdBase.sendFloat("sound_slider", progress.toFloat())
                    decay -> PdBase.sendFloat("decay_slider", progress.toFloat())
                    speed -> PdBase.sendFloat("bmp_slider", progress.toFloat())
                    mld_volume -> PdBase.sendFloat("melody_volume", progress.toFloat())
                    chord_volume -> PdBase.sendFloat("chord_volume", progress.toFloat())
                    attack -> PdBase.sendFloat("chord_attack_slider", progress.toFloat())
                }
            }
        }
        pitch.setOnSeekBarChangeListener(listener)
        dark.setOnSeekBarChangeListener(listener)
        decay.setOnSeekBarChangeListener(listener)
        speed.setOnSeekBarChangeListener(listener)
        mld_volume.setOnSeekBarChangeListener(listener)
        chord_volume.setOnSeekBarChangeListener(listener)
        attack.setOnSeekBarChangeListener(listener)

        btn_next_chord.setOnClickListener { PdBase.sendBang("change_chord") }
//        PdBase.sendFloat("play", 1f)
//        PdBase.sendFloat("master_volume", 70F)
    }

    abstract class SimpleSeekBarListener : SeekBar.OnSeekBarChangeListener {
        override fun onStartTrackingTouch(seekBar: SeekBar?) = Unit
        override fun onStopTrackingTouch(seekBar: SeekBar?) = Unit
    }

    private lateinit var displayMetrics: DisplayMetrics

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        displayMetrics = DisplayMetrics()
        (context as Activity).windowManager
            .defaultDisplay
            .getMetrics(displayMetrics)
        try {
            initPd()
        } catch (e: IOException) {
        }
    }

    @Throws(IOException::class)
    private fun initPd() {
        AudioParameters.init(context)
        val srate = max(MIN_SAMPLE_RATE, AudioParameters.suggestSampleRate())
        PdAudio.initAudio(srate, 0, 2, 1, true)
        val dir = activity?.filesDir
        val patchFile = File(dir, "main.pd")
        IoUtils.extractZipResource(resources.openRawResource(R.raw.patch), dir, true)
        PdBase.openPatch(patchFile.absolutePath)
    }

    override fun onResume() {
        super.onResume()
        startBackgroundThread()
    }

    override fun onStart() {
        super.onStart()
        openCamera()
        posenet = Posenet(this.context!!)
        PdAudio.startAudio(context)
    }

    override fun onStop() {
        PdAudio.stopAudio()
        super.onStop()
    }

    override fun onPause() {
        closeCamera()
        stopBackgroundThread()
        super.onPause()
    }

    private fun cleanup() { // make sure to release all resources
        PdAudio.release()
        PdBase.release()
    }

    override fun onDestroy() {
        super.onDestroy()
        cleanup()
        posenet.close()
    }

    private fun requestCameraPermission() {
        if (shouldShowRequestPermissionRationale(Manifest.permission.CAMERA)) {
            ConfirmationDialog().show(childFragmentManager, FRAGMENT_DIALOG)
        } else {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), REQUEST_CAMERA_PERMISSION)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (allPermissionsGranted(grantResults)) {
                ErrorDialog.newInstance(getString(R.string.tfe_pn_request_permission))
                    .show(childFragmentManager, FRAGMENT_DIALOG)
            }
        } else {
            super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        }
    }

    private fun allPermissionsGranted(grantResults: IntArray) = grantResults.all {
        it == PackageManager.PERMISSION_GRANTED
    }

    /**
     * Sets up member variables related to camera.
     */
    private fun setUpCameraOutputs() {

        val activity = activity
        val manager = activity!!.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        try {
            for (cameraId in manager.cameraIdList) {
                val characteristics = manager.getCameraCharacteristics(cameraId)

                // We don't use a front facing camera in this sample.
                val cameraDirection = characteristics.get(CameraCharacteristics.LENS_FACING)
                if (cameraDirection != null &&
                    cameraDirection == CameraCharacteristics.LENS_FACING_FRONT
                ) {
                    continue
                }

                previewSize = Size(PREVIEW_WIDTH, PREVIEW_HEIGHT)

                imageReader = ImageReader.newInstance(
                    PREVIEW_WIDTH, PREVIEW_HEIGHT,
                    ImageFormat.YUV_420_888, /*maxImages*/ 2
                )

                sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION)!!

                previewHeight = previewSize!!.height
                previewWidth = previewSize!!.width

                // Initialize the storage bitmaps once when the resolution is known.
                rgbBytes = IntArray(previewWidth * previewHeight)

                // Check if the flash is supported.
                flashSupported =
                    characteristics.get(CameraCharacteristics.FLASH_INFO_AVAILABLE) == true

                this.cameraId = cameraId

                // We've found a viable camera and finished setting up member variables,
                // so we don't need to iterate through other available cameras.
                return
            }
        } catch (e: CameraAccessException) {
            Log.e(TAG, e.toString())
        } catch (e: NullPointerException) {
            // Currently an NPE is thrown when the Camera2API is used but not supported on the
            // device this code runs.
            ErrorDialog.newInstance(getString(R.string.tfe_pn_camera_error))
                .show(childFragmentManager, FRAGMENT_DIALOG)
        }
    }

    /**
     * Opens the camera specified by [PosenetFragment.cameraId].
     */
    private fun openCamera() {
        val permissionCamera =
            ContextCompat.checkSelfPermission(activity!!, Manifest.permission.CAMERA)
        if (permissionCamera != PackageManager.PERMISSION_GRANTED) {
            requestCameraPermission()
        }
        setUpCameraOutputs()
        val manager = activity!!.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        try {
            // Wait for camera to open - 2.5 seconds is sufficient
            if (!cameraOpenCloseLock.tryAcquire(2500, TimeUnit.MILLISECONDS)) {
                throw RuntimeException("Time out waiting to lock camera opening.")
            }
            manager.openCamera(cameraId!!, stateCallback, backgroundHandler)
        } catch (e: CameraAccessException) {
            Log.e(TAG, e.toString())
        } catch (e: InterruptedException) {
            throw RuntimeException("Interrupted while trying to lock camera opening.", e)
        }
    }

    /**
     * Closes the current [CameraDevice].
     */
    private fun closeCamera() {
        if (captureSession == null) {
            return
        }

        try {
            cameraOpenCloseLock.acquire()
            captureSession!!.close()
            captureSession = null
            cameraDevice!!.close()
            cameraDevice = null
            imageReader!!.close()
            imageReader = null
        } catch (e: InterruptedException) {
            throw RuntimeException("Interrupted while trying to lock camera closing.", e)
        } finally {
            cameraOpenCloseLock.release()
        }
    }

    /**
     * Starts a background thread and its [Handler].
     */
    private fun startBackgroundThread() {
        backgroundThread = HandlerThread("imageAvailableListener").also { it.start() }
        backgroundHandler = Handler(backgroundThread!!.looper)
    }

    /**
     * Stops the background thread and its [Handler].
     */
    private fun stopBackgroundThread() {
        backgroundThread?.quitSafely()
        try {
            backgroundThread?.join()
            backgroundThread = null
            backgroundHandler = null
        } catch (e: InterruptedException) {
            Log.e(TAG, e.toString())
        }
    }

    /** Fill the yuvBytes with data from image planes.   */
    private fun fillBytes(planes: Array<Image.Plane>, yuvBytes: Array<ByteArray?>) {
        // Row stride is the total number of bytes occupied in memory by a row of an image.
        // Because of the variable row stride it's not possible to know in
        // advance the actual necessary dimensions of the yuv planes.
        for (i in planes.indices) {
            val buffer = planes[i].buffer
            if (yuvBytes[i] == null) {
                yuvBytes[i] = ByteArray(buffer.capacity())
            }
            buffer.get(yuvBytes[i]!!)
        }
    }

    /** A [OnImageAvailableListener] to receive frames as they are available.  */
    private var imageAvailableListener = object : OnImageAvailableListener {
        override fun onImageAvailable(imageReader: ImageReader) {
            // We need wait until we have some size from onPreviewSizeChosen
            if (previewWidth == 0 || previewHeight == 0) {
                return
            }

            val image = imageReader.acquireLatestImage() ?: return
            fillBytes(image.planes, yuvBytes)

            ImageUtils.convertYUV420ToARGB8888(
                yuvBytes[0]!!,
                yuvBytes[1]!!,
                yuvBytes[2]!!,
                previewWidth,
                previewHeight,
                /*yRowStride=*/ image.planes[0].rowStride,
                /*uvRowStride=*/ image.planes[1].rowStride,
                /*uvPixelStride=*/ image.planes[1].pixelStride,
                rgbBytes
            )

            // Create bitmap from int array
            val imageBitmap = Bitmap.createBitmap(
                rgbBytes, previewWidth, previewHeight,
                Bitmap.Config.ARGB_8888
            )

            // Create rotated version for portrait display
            val rotateMatrix = Matrix()
            rotateMatrix.postRotate(90.0f)

            val rotatedBitmap = Bitmap.createBitmap(
                imageBitmap, 0, 0, previewWidth, previewHeight,
                rotateMatrix, true
            )
            image.close()

            // Process an image for analysis in every 3 frames.
            frameCounter = (frameCounter + 1) % 3
            if (frameCounter == 0) {
                processImage(rotatedBitmap)
            }
        }
    }

    /** Crop Bitmap to maintain aspect ratio of model input.   */
    private fun cropBitmap(bitmap: Bitmap): Bitmap {
        val bitmapRatio = bitmap.height.toFloat() / bitmap.width
        val modelInputRatio = MODEL_HEIGHT.toFloat() / MODEL_WIDTH
        var croppedBitmap = bitmap

        // Acceptable difference between the modelInputRatio and bitmapRatio to skip cropping.
        val maxDifference = 1e-5

        // Checks if the bitmap has similar aspect ratio as the required model input.
        when {
            abs(modelInputRatio - bitmapRatio) < maxDifference -> return croppedBitmap
            modelInputRatio < bitmapRatio -> {
                // New image is taller so we are height constrained.
                val cropHeight = bitmap.height - (bitmap.width.toFloat() / modelInputRatio)
                croppedBitmap = Bitmap.createBitmap(
                    bitmap,
                    0,
                    (cropHeight / 2).toInt(),
                    bitmap.width,
                    (bitmap.height - cropHeight).toInt()
                )
            }
            else -> {
                val cropWidth = bitmap.width - (bitmap.height.toFloat() * modelInputRatio)
                croppedBitmap = Bitmap.createBitmap(
                    bitmap,
                    (cropWidth / 2).toInt(),
                    0,
                    (bitmap.width - cropWidth).toInt(),
                    bitmap.height
                )
            }
        }
        return croppedBitmap
    }

    /** Set the paint color and size.    */
    private fun setPaint() {
        paint.color = Color.WHITE
        paint.textSize = 20.0f
        paint.strokeWidth = 5.0f
    }

    /** Draw bitmap on Canvas.   */
    private fun draw(canvas: Canvas, person: Person, bitmap: Bitmap) {
        canvas.drawColor(Color.BLACK, PorterDuff.Mode.CLEAR)
        // Draw `bitmap` and `person` in square canvas.
        val screenWidth: Int
        val screenHeight: Int
        val left: Int
        val right: Int
        val top: Int
        val bottom: Int
        if (canvas.height > canvas.width) {
            screenWidth = canvas.width
            screenHeight = canvas.width
            left = 0
            top = (canvas.height - canvas.width) / 2
        } else {
            screenWidth = canvas.height
            screenHeight = canvas.height
            left = (canvas.width - canvas.height) / 2
            top = 0
        }

        setPaint()

        val widthRatio = screenWidth.toFloat() / MODEL_WIDTH
        val heightRatio = screenHeight.toFloat() / MODEL_HEIGHT

        // Draw key points over the image.
        for (keyPoint in person.keyPoints) {
            if (keyPoint.score > minConfidence && trackingBodyParts.contains(keyPoint.bodyPart)) {
                val position = keyPoint.position
                val adjustedX: Float = position.x.toFloat() * widthRatio + left
                val adjustedY: Float = position.y.toFloat() * heightRatio + top
                canvas.drawCircle(adjustedX, adjustedY, circleRadius, paint)
            }
        }

        for (line in bodyJoints) {
            if (
                (person.keyPoints[line.first.ordinal].score > minConfidence) and
                (person.keyPoints[line.second.ordinal].score > minConfidence)
            ) {
                canvas.drawLine(
                    person.keyPoints[line.first.ordinal].position.x.toFloat() * widthRatio + left,
                    person.keyPoints[line.first.ordinal].position.y.toFloat() * heightRatio + top,
                    person.keyPoints[line.second.ordinal].position.x.toFloat() * widthRatio + left,
                    person.keyPoints[line.second.ordinal].position.y.toFloat() * heightRatio + top,
                    paint
                )
            }
        }

        // Draw!
        surfaceHolder!!.unlockCanvasAndPost(canvas)
    }

    /** Process image using Posenet library.   */
    private fun processImage(bitmap: Bitmap) {
        // Crop bitmap.
        val croppedBitmap = cropBitmap(bitmap)

        // Created scaled version of bitmap for model input.
        val scaledBitmap = Bitmap.createScaledBitmap(croppedBitmap, MODEL_WIDTH, MODEL_HEIGHT, true)

        // Perform inference.
        val person = posenet.estimateSinglePose(scaledBitmap)
//        if (highConfidenceInPose(person)) {
            val lEye = person.keyPoints[BodyPart.LEFT_EYE.ordinal]
            val lWrist = person.keyPoints[BodyPart.LEFT_WRIST.ordinal]
            val lElbow = person.keyPoints[BodyPart.LEFT_ELBOW.ordinal]
            val lShoulder = person.keyPoints[BodyPart.LEFT_SHOULDER.ordinal]
            val lHip = person.keyPoints[BodyPart.LEFT_HIP.ordinal]
            val lEyeToShoulder = abs(lEye.position.x - lShoulder.position.x)
            val lShoulderToElbow = abs(lShoulder.position.x - lElbow.position.x)

            val rEye = person.keyPoints[BodyPart.RIGHT_EYE.ordinal]
            val rWrist = person.keyPoints[BodyPart.RIGHT_WRIST.ordinal]
            val rElbow = person.keyPoints[BodyPart.RIGHT_ELBOW.ordinal]
            val rShoulder = person.keyPoints[BodyPart.RIGHT_SHOULDER.ordinal]
            val rHip = person.keyPoints[BodyPart.RIGHT_HIP.ordinal]
            val rEyeToShoulder = abs(rShoulder.position.x - rEye.position.x)
            val rShoulderToElbow = abs(rElbow.position.x - rShoulder.position.x)
            Log.i(TAG, "lEyeToShoulder: $lEyeToShoulder; rEyeToShoulder: $rEyeToShoulder")
            Log.i(TAG, "lShoulderToElbow: $lShoulderToElbow; rShoulderToElbow: $rShoulderToElbow\"")
            val lSide = lEyeToShoulder + lShoulderToElbow > rEyeToShoulder + rShoulderToElbow
            val armLength = abs(lShoulder.position.x - rShoulder.position.x) * 2
            if (lSide) {
                lSideConfidence++
            } else {
                rSideConfidence++
            }

            if (lSideConfidence + rSideConfidence > 3) {
                currentSide = if (lSideConfidence > rSideConfidence) LEFT else RIGHT
                Log.i(TAG, "currentSide CHANGE: $currentSide")
                lSideConfidence = 0
                rSideConfidence = 0
            }
            Log.i(TAG, "currentSide: $currentSide")
            when (currentSide) {
                LEFT -> {
                    stopPd(lHip, lWrist)
                    handSonification(lWrist, lElbow, lShoulder, armLength)
                }
                RIGHT -> {
                    stopPd(rHip, rWrist)
                    handSonification(
                        rWrist,
                        rElbow,
                        rShoulder,
                        armLength
                    ) // left is right to the phone
                }
            }
//        }

        val canvas: Canvas = surfaceHolder!!.lockCanvas()
        draw(canvas, person, scaledBitmap)
    }

    private fun stopPd(
        hip: KeyPoint,
        wrist: KeyPoint
    ) {
        if (hip.position.y - wrist.position.y < 0) {
            stopConfidence++
            if (stopConfidence > 3) {
                stopConfidence = 0
                PdBase.sendFloat("play", 0f)
                startPlayingTime = 0
                playing = false
            }
        }
    }

    private fun handSonification(
        wrist: KeyPoint,
        elbow: KeyPoint,
        shoulder: KeyPoint,
        armLength: Int
    ) {
        if (
            wrist.score > minConfidence &&
            elbow.score > minConfidence &&
            shoulder.score > minConfidence
        ) {
            var scale = 127 + (shoulder.position.x - wrist.position.x) * (127 / armLength)
            scale = max(0, min(127, scale))
//            PdBase.sendFloat("bang_scale", scale.toFloat())

            val avrgX = (wrist.position.x + shoulder.position.x) / 2

            val x = abs(wrist.position.x - avrgX) + abs(shoulder.position.x - avrgX)

            val avrgY = (wrist.position.y + shoulder.position.y) / 2

            val y = abs(wrist.position.y - avrgY) + abs(shoulder.position.y - avrgY)

            val arm = max(127, max(abs(shoulder.position.y - wrist.position.y), armLength))
            var newVolume = (64 + (shoulder.position.y - wrist.position.y) * (127 / armLength))
            newVolume = max(40, min(127, newVolume))
            PdBase.sendFloat("master_volume", newVolume.toFloat())

            if (volumeAnimator.isRunning) volumeAnimator.cancel()

            volumeAnimator.setFloatValues(volume, newVolume.toFloat())
            volumeAnimator.start()

            if (x < armLength / 2 && y < armLength / 2 && !playing) {
                PdBase.sendFloat("play", 1f)
                displayMetrics.heightPixels / 127
                startPlayingTime = System.currentTimeMillis()
                playing = true
            }
        }
    }

    private fun highConfidenceInPose(person: Person): Boolean {
        return BodyPart.values()
            .takeWhile { it != BodyPart.LEFT_KNEE }
            .map { person.keyPoints[it.ordinal] }
            .all { it.score > minConfidence }
    }

    /**
     * Creates a new [CameraCaptureSession] for camera preview.
     */
    private fun createCameraPreviewSession() {
        try {

            // We capture images from preview in YUV format.
            imageReader = ImageReader.newInstance(
                previewSize!!.width, previewSize!!.height, ImageFormat.YUV_420_888, 2
            )
            imageReader!!.setOnImageAvailableListener(imageAvailableListener, backgroundHandler)

            // This is the surface we need to record images for processing.
            val recordingSurface = imageReader!!.surface

            // We set up a CaptureRequest.Builder with the output Surface.
            previewRequestBuilder = cameraDevice!!.createCaptureRequest(
                CameraDevice.TEMPLATE_PREVIEW
            )
            previewRequestBuilder!!.addTarget(recordingSurface)

            // Here, we create a CameraCaptureSession for camera preview.
            cameraDevice!!.createCaptureSession(
                listOf(recordingSurface),
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(cameraCaptureSession: CameraCaptureSession) {
                        // The camera is already closed
                        if (cameraDevice == null) return

                        // When the session is ready, we start displaying the preview.
                        captureSession = cameraCaptureSession
                        try {
                            // Auto focus should be continuous for camera preview.
                            previewRequestBuilder!!.set(
                                CaptureRequest.CONTROL_AF_MODE,
                                CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE
                            )
                            // Flash is automatically enabled when necessary.
                            setAutoFlash(previewRequestBuilder!!)

                            // Finally, we start displaying the camera preview.
                            previewRequest = previewRequestBuilder!!.build()
                            captureSession!!.setRepeatingRequest(
                                previewRequest!!,
                                captureCallback, backgroundHandler
                            )
                        } catch (e: CameraAccessException) {
                            Log.e(TAG, e.toString())
                        }
                    }

                    override fun onConfigureFailed(cameraCaptureSession: CameraCaptureSession) {
                        showToast("Failed")
                    }
                },
                null
            )
        } catch (e: CameraAccessException) {
            Log.e(TAG, e.toString())
        }
    }

    private fun setAutoFlash(requestBuilder: CaptureRequest.Builder) {
        if (flashSupported) {
            requestBuilder.set(
                CaptureRequest.CONTROL_AE_MODE,
                CaptureRequest.CONTROL_AE_MODE_ON_AUTO_FLASH
            )
        }
    }

    /**
     * Shows an error message dialog.
     */
    class ErrorDialog : DialogFragment() {

        override fun onCreateDialog(savedInstanceState: Bundle?): Dialog =
            AlertDialog.Builder(activity)
                .setMessage(arguments!!.getString(ARG_MESSAGE))
                .setPositiveButton(android.R.string.ok) { _, _ -> activity!!.finish() }
                .create()

        companion object {

            @JvmStatic
            private val ARG_MESSAGE = "message"

            @JvmStatic
            fun newInstance(message: String): ErrorDialog = ErrorDialog().apply {
                arguments = Bundle().apply { putString(ARG_MESSAGE, message) }
            }
        }
    }

    companion object {
        /**
         * Conversion from screen rotation to JPEG orientation.
         */
        private val ORIENTATIONS = SparseIntArray()
        private val FRAGMENT_DIALOG = "dialog"

        private const val NONE = -1
        private const val LEFT = 0
        private const val RIGHT = 1

        init {
            ORIENTATIONS.append(Surface.ROTATION_0, 90)
            ORIENTATIONS.append(Surface.ROTATION_90, 0)
            ORIENTATIONS.append(Surface.ROTATION_180, 270)
            ORIENTATIONS.append(Surface.ROTATION_270, 180)
        }

        private const val MIN_SAMPLE_RATE = 44100

        /**
         * Tag for the [Log].
         */
        private const val TAG = "PosenetActivity"
    }
}
