package com.direwolf.archeryhelper.activities

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.BitmapFactory.decodeResource
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.direwolf.archeryhelper.R
import com.direwolf.archeryhelper.image.CapturedImageRepository
import com.direwolf.archeryhelper.utils.debugLog
import java.io.File

class ScanActivity : TemplateActivity() {
    override fun getLayoutId(): Int = R.layout.activity_scan

    private lateinit var previewView: PreviewView
    private lateinit var photoView: ImageView
    private lateinit var btnScan: Button
    private var imageCapture: ImageCapture? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        previewView = findViewById(R.id.previewView)
        photoView = findViewById(R.id.photoView)
        btnScan = findViewById(R.id.btnScan)

//        debug()

        if (!allPermissionsGranted()) {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        } else {
            startCamera()
        }

        btnScan.setOnClickListener {
            takePhoto()
        }

        findViewById<Button>(R.id.btnContinue).setOnClickListener {
        startActivity(Intent(this, EditActivity::class.java))
            finish()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS && allPermissionsGranted()) {
            startCamera()
        } else if (requestCode == REQUEST_CODE_PERMISSIONS) {
            Toast.makeText(this, "Нет доступа к камере", Toast.LENGTH_SHORT).show()
        }
    }

    private fun debug() {
        val bitmap = decodeResource(resources, R.drawable.test)
        val image = cropToSquare(bitmap)
        photoView.setImageBitmap(image)
        photoView.visibility = ImageView.VISIBLE
        previewView.visibility = PreviewView.GONE
        findViewById<Button>(R.id.btnContinue).isEnabled = true
        CapturedImageRepository.save(this, image)
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    imageCapture
                )
            } catch (e: Exception) {
                debugLog(e.message ?: "Камера не запущена")
                Toast.makeText(this, "Камера не запущена", Toast.LENGTH_SHORT).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun takePhoto() {
        val capture = imageCapture ?: return
        val file = File(cacheDir, "camera_capture.jpg")
        val outputOptions = ImageCapture.OutputFileOptions.Builder(file).build()
        capture.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                    val bitmap = BitmapFactory.decodeFile(file.absolutePath)
                    if (bitmap == null) {
                        Toast.makeText(this@ScanActivity, "Фото не получено", Toast.LENGTH_SHORT).show()
                        return
                    }
                    val image = cropToSquare(bitmap)
                    photoView.setImageBitmap(image)
                    photoView.visibility = ImageView.VISIBLE
                    previewView.visibility = PreviewView.GONE
                    findViewById<Button>(R.id.btnContinue).isEnabled = true
                    CapturedImageRepository.save(this@ScanActivity, image)
                }

                override fun onError(exception: ImageCaptureException) {
                    debugLog(exception.message ?: "Фото не получено")
                    Toast.makeText(this@ScanActivity, "Фото не получено", Toast.LENGTH_SHORT).show()
                }
            }
        )
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private fun cropToSquare(bitmap: Bitmap): Bitmap {
        val dimension = minOf(bitmap.width, bitmap.height)
        val x = (bitmap.width - dimension) / 2
        val y = (bitmap.height - dimension) / 2
        return Bitmap.createBitmap(bitmap, x, y, dimension, dimension)
    }

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}
