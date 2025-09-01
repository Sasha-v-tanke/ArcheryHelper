package com.direwolf.archeryhelper.activities

import android.content.Context
import android.graphics.*
import android.os.Bundle
import android.view.MotionEvent
import android.widget.*
import com.direwolf.archeryhelper.R
import com.direwolf.archeryhelper.managers.Application
import com.direwolf.archeryhelper.utils.debugLog
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.sin
import kotlin.math.sqrt

class EditActivity : TemplateActivity() {
    override fun getLayoutId(): Int = R.layout.activity_edit

    private lateinit var imageView: ImageView
    private lateinit var module: Module

    // UI-кнопки
    private lateinit var btnEdit: Button
    private lateinit var btnAdd: Button
    private lateinit var btnRemove: Button
    private lateinit var btnPrev: Button
    private lateinit var btnNext: Button
    private lateinit var btnBack: Button
    private lateinit var btnContinue: Button

    private val points = mutableListOf<Pair<Float, Float>>()
    private var selectedIndex = -1
    private var editMode = false
    private var maxRadius: Float = 0f
    private var addMode = false

    override fun onPostCreate(savedInstanceState: Bundle?) {
        super.onPostCreate(savedInstanceState)
        imageView.post {
            var flag = true
            val bitmap = (application as Application).imageHolder.getImage()
            if (bitmap == null) {
                Toast.makeText(this, "Нет фото для анализа", Toast.LENGTH_SHORT).show()
                finish()
                flag = false
            }
            if (flag) {
                module = LiteModuleLoader.load(assetFilePath(this, "model.ptl"))

                val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
                val std = floatArrayOf(0.229f, 0.224f, 0.225f)
                val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap!!, mean, std)

                val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()
                val scores = outputTensor.dataAsFloatArray
                debugLog("Выход модели: ${scores.joinToString()}")

                for (i in scores.indices step 2) {
                    val r = scores[i]
                    val theta = scores[i + 1]
//                    debugLog("$r $theta")
                    if (r in 0.0..1.0 && theta >= 0) {
                        points.add(Pair(r, theta))
                    }
                }

                redraw()
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        imageView = findViewById(R.id.imageView)

        btnEdit = findViewById(R.id.btnEdit)
        btnAdd = findViewById(R.id.btnAdd)
        btnRemove = findViewById(R.id.btnRemove)
        btnPrev = findViewById(R.id.btnPrev)
        btnNext = findViewById(R.id.btnNxt)
        btnBack = findViewById(R.id.btnBack)
        btnContinue = findViewById(R.id.btnContinue)

        setEditButtonsVisible(false)

        btnEdit.setOnClickListener {
            editMode = !editMode
            setEditButtonsVisible(editMode)
            if (!editMode) {
                selectedIndex = -1
                addMode = false
            } else if (points.isNotEmpty()) {
                selectedIndex = 0
            }
            redraw()
        }

        btnNext.setOnClickListener {
            if (points.isNotEmpty()) {
                selectedIndex = (selectedIndex + 1) % points.size
                redraw()
            }
        }

        btnPrev.setOnClickListener {
            if (points.isNotEmpty()) {
                selectedIndex = if (selectedIndex - 1 < 0) points.size - 1 else selectedIndex - 1
                redraw()
            }
        }

        btnAdd.setOnClickListener {
            addMode = !addMode
            btnAdd.text = if (addMode) "Отмена" else "Добавить"
//            Toast.makeText(
//                this,
//                if (addMode) "Тапните по экрану для добавления точки" else "Режим добавления выключен",
//                Toast.LENGTH_SHORT
//            ).show()
        }

        btnRemove.setOnClickListener {
            if (selectedIndex in points.indices) {
                points.removeAt(selectedIndex)
                if (points.isNotEmpty()) {
                    selectedIndex %= points.size
                } else {
                    selectedIndex = -1
                }
                redraw()
            }
        }

        btnBack.setOnClickListener { finish() }

        btnContinue.setOnClickListener {
            Toast.makeText(this, "Продолжить (пока пусто)", Toast.LENGTH_SHORT).show()
        }

        // обработка тапа по картинке
        imageView.setOnTouchListener { _, event ->
            if (editMode && addMode && event.action == MotionEvent.ACTION_DOWN && maxRadius != 0f) {
                val x = event.x
                val y = event.y

                // координаты в центр и радиус+угол
                val cx = maxRadius
                val cy = maxRadius
                val dx = x - cx
                val dy = y - cy
                var r = sqrt(dx * dx + dy * dy) / maxRadius
                var theta = Math.toDegrees(kotlin.math.atan2(dy, dx).toDouble()).toFloat()
                if (theta < 0) theta += 360f
                debugLog("$r, $theta")

                points.add(Pair(r, theta))
                selectedIndex = points.size - 1
                addMode = false
                btnAdd.text = "Добавить"
                redraw()
            }
            true
        }
    }

    private fun setEditButtonsVisible(visible: Boolean) {
        val v = if (visible) Button.VISIBLE else Button.GONE
        btnAdd.visibility = v
        btnRemove.visibility = v
        btnPrev.visibility = v
        btnNext.visibility = v
    }

    private fun redraw() {
        val bitmap = BitmapFactory.decodeResource(resources, R.drawable.target)
        val scaled = Bitmap.createScaledBitmap(bitmap, 800, 800, true)
        val copy = scaled.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(copy)

        val paintNormal = Paint().apply {
            color = Color.GREEN
            style = Paint.Style.FILL
            strokeWidth = 10f
        }
        val paintSelected = Paint().apply {
            color = Color.RED
            style = Paint.Style.FILL
            strokeWidth = 12f
        }

        maxRadius = copy.width / 2f
//        debugLog(maxRadius.toString())
        val cx = maxRadius
        val cy = maxRadius
        for ((i, point) in points.withIndex()) {
            val (r, theta) = point
            val x = cx + r * cos(theta / 180f * PI) * maxRadius
            val y = cy + r * sin(theta / 180f * PI) * maxRadius
            val paint = if (i == selectedIndex) paintSelected else paintNormal
            canvas.drawCircle(x.toFloat(), y.toFloat(), 12f, paint)
        }

        imageView.setImageBitmap(copy)
    }

    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
//        if (!file.exists() || file.length() == 0L) {
        context.assets.open(assetName).use { input ->
            FileOutputStream(file).use { output -> input.copyTo(output) }
        }
//        }
        return file.absolutePath
    }
}