package com.direwolf.archeryhelper.activities

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.direwolf.archeryhelper.R
import com.direwolf.archeryhelper.managers.Application
import com.direwolf.archeryhelper.utils.debugLog

class ScanActivity : TemplateActivity() {
    override fun getLayoutId(): Int = R.layout.activity_scan

    private lateinit var photoView: ImageView
    private lateinit var btnScan: Button
    private var image: Bitmap? = null

    private val cameraLauncher =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == RESULT_OK) {
                val imageBitmap = result.data?.extras?.get("data") as? Bitmap
                if (imageBitmap != null) {
                    image = cropToSquare(imageBitmap)
                    photoView.setImageBitmap(image)
                    findViewById<Button>(R.id.btnContinue).isEnabled = true
                    (application as Application).imageHolder.setImage(image!!)
                } else {
                    debugLog("Фото не получено")
                }
            } else {
                debugLog("Камера отменена")
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        photoView = findViewById(R.id.photoView)
        btnScan = findViewById(R.id.btnScan)

        if (!allPermissionsGranted()) {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        btnScan.setOnClickListener {
            openCameraApp()
        }
    }

    private fun openCameraApp() {
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        cameraLauncher.launch(intent)
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