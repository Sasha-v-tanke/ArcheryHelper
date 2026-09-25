package com.direwolf.archeryhelper.activities

import android.content.Intent
import android.os.Bundle
import android.view.MotionEvent
import android.widget.*
import androidx.lifecycle.lifecycleScope
import com.direwolf.archeryhelper.R
import com.direwolf.archeryhelper.domain.ShotEditor
import com.direwolf.archeryhelper.domain.ShotPoint
import com.direwolf.archeryhelper.image.CapturedImageRepository
import com.direwolf.archeryhelper.managers.DataManager
import com.direwolf.archeryhelper.ml.TorchShotDetector
import com.direwolf.archeryhelper.stats.ScoreCalculator
import com.direwolf.archeryhelper.ui.TargetOverlayRenderer
import com.direwolf.archeryhelper.utils.Series
import com.direwolf.archeryhelper.utils.Shot
import com.direwolf.archeryhelper.utils.debugLog
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.PI
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

class EditActivity : TemplateActivity() {
    override fun getLayoutId(): Int = R.layout.activity_edit

    private lateinit var imageView: ImageView
    private lateinit var shotDetector: TorchShotDetector

    // UI-кнопки
    private lateinit var btnEdit: Button
    private lateinit var btnAdd: Button
    private lateinit var btnRemove: Button
    private lateinit var btnUndo: Button
    private lateinit var btnPrev: Button
    private lateinit var btnNext: Button
    private lateinit var btnReset: Button
    private lateinit var btnBack: Button
    private lateinit var btnContinue: Button

    private var shotEditor = ShotEditor()
    private var selectedIndex = -1
    private var editMode = false
    private var maxRadius: Float = 0f
    private var addMode = false
    private var dragMode = false
    private var restoredState = false

    override fun onPostCreate(savedInstanceState: Bundle?) {
        super.onPostCreate(savedInstanceState)
        imageView.post {
            if (restoredState) {
                redraw()
                return@post
            }
            val bitmap = CapturedImageRepository.load(this)
            if (bitmap == null) {
                Toast.makeText(this, "Нет фото для анализа", Toast.LENGTH_SHORT).show()
                finish()
                return@post
            }

            lifecycleScope.launch {
                try {
                    val detections = withContext(Dispatchers.Default) {
                        shotDetector.detect(bitmap)
                    }
                    shotEditor = ShotEditor(detections.map { polarToShotPoint(it.radiusNorm, it.angleDeg) })
                    redraw()
                } catch (e: Exception) {
                    debugLog(e.message ?: "Ошибка распознавания")
                    Toast.makeText(this@EditActivity, "Ошибка распознавания", Toast.LENGTH_SHORT).show()
                    redraw()
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        imageView = findViewById(R.id.imageView)
        shotDetector = TorchShotDetector(this)

        btnEdit = findViewById(R.id.btnEdit)
        btnAdd = findViewById(R.id.btnAdd)
        btnRemove = findViewById(R.id.btnRemove)
        btnUndo = findViewById(R.id.btnUndo)
        btnPrev = findViewById(R.id.btnPrev)
        btnNext = findViewById(R.id.btnNxt)
        btnReset = findViewById(R.id.btnReset)
        btnBack = findViewById(R.id.btnBack)
        btnContinue = findViewById(R.id.btnContinue)

        setEditButtonsVisible(false)
        restoreState(savedInstanceState)

        btnEdit.setOnClickListener {
            editMode = !editMode
            setEditButtonsVisible(editMode)
            if (!editMode) {
                selectedIndex = -1
                addMode = false
                dragMode = false
            } else if (shotEditor.snapshot().isNotEmpty()) {
                selectedIndex = 0
            }
            redraw()
        }

        btnNext.setOnClickListener {
            val points = shotEditor.snapshot()
            if (points.isNotEmpty()) {
                selectedIndex = (selectedIndex + 1) % points.size
                redraw()
            }
        }

        btnPrev.setOnClickListener {
            val points = shotEditor.snapshot()
            if (points.isNotEmpty()) {
                selectedIndex = if (selectedIndex - 1 < 0) points.size - 1 else selectedIndex - 1
                redraw()
            }
        }

        btnAdd.setOnClickListener {
            addMode = !addMode
            btnAdd.text = if (addMode) "Отмена" else "Добавить"
            Toast.makeText(
                this,
                if (addMode) "Тапните по экрану для добавления точки" else "Режим добавления выключен",
                Toast.LENGTH_SHORT
            ).show()
        }

        btnRemove.setOnClickListener {
            val points = shotEditor.snapshot()
            if (selectedIndex in points.indices) {
                shotEditor.remove(selectedIndex)
                val updatedPoints = shotEditor.snapshot()
                if (updatedPoints.isNotEmpty()) {
                    selectedIndex %= updatedPoints.size
                } else {
                    selectedIndex = -1
                }
                redraw()
            }
        }

        btnUndo.setOnClickListener {
            if (shotEditor.undo()) {
                normalizeSelectedIndex()
                redraw()
            }
        }

        btnReset.setOnClickListener {
            shotEditor.reset()
            normalizeSelectedIndex()
            redraw()
        }

        btnBack.setOnClickListener {
            startActivity(Intent(this, ScanActivity::class.java))
            finish()
        }

        btnContinue.setOnClickListener {
            val shots = mutableListOf<Shot>()
            val points = shotEditor.snapshot()
            for (i in points.indices) {
                val radius = points[i].radiusNorm()
                shots.add(Shot(i + 1, parseRadius(radius), radius, angleDeg(points[i])))
            }
            val series = Series(DataManager.getLastSeriesIndex() + 1, shots)
            DataManager.saveSeries(series, DataManager.getLastDistanceIndex())
            finish()
        }

        imageView.setOnTouchListener { _, event ->
            if (editMode && maxRadius != 0f) {
                when (event.action) {
                    MotionEvent.ACTION_DOWN -> handleTouchDown(event.x, event.y)
                    MotionEvent.ACTION_MOVE -> handleTouchMove(event.x, event.y)
                    MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> dragMode = false
                }
            }
            true
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        val points = shotEditor.snapshot()
        outState.putFloatArray(STATE_X_POINTS, points.map { it.xNorm }.toFloatArray())
        outState.putFloatArray(STATE_Y_POINTS, points.map { it.yNorm }.toFloatArray())
        outState.putInt(STATE_SELECTED_INDEX, selectedIndex)
        outState.putBoolean(STATE_EDIT_MODE, editMode)
        outState.putBoolean(STATE_ADD_MODE, addMode)
    }

    private fun parseRadius(radius: Float): Int {
        return ScoreCalculator.scoreRadius(radius)
    }

    private fun restoreState(savedInstanceState: Bundle?) {
        if (savedInstanceState == null) return
        val xs = savedInstanceState.getFloatArray(STATE_X_POINTS) ?: return
        val ys = savedInstanceState.getFloatArray(STATE_Y_POINTS) ?: return
        if (xs.size != ys.size) return

        shotEditor = ShotEditor(xs.indices.map { ShotPoint(xs[it], ys[it]) })
        selectedIndex = savedInstanceState.getInt(STATE_SELECTED_INDEX, -1)
        editMode = savedInstanceState.getBoolean(STATE_EDIT_MODE, false)
        addMode = savedInstanceState.getBoolean(STATE_ADD_MODE, false)
        restoredState = true
        setEditButtonsVisible(editMode)
        btnAdd.text = if (addMode) "Отмена" else "Добавить"
        normalizeSelectedIndex()
    }

    private fun setEditButtonsVisible(visible: Boolean) {
        val v = if (visible) Button.VISIBLE else Button.GONE
        btnAdd.visibility = v
        btnRemove.visibility = v
        btnUndo.visibility = v
        btnPrev.visibility = v
        btnNext.visibility = v
        btnReset.visibility = v
        btnEdit.text = if (visible) "Сохранить" else "Редактировать"
    }

    private fun redraw() {
        val copy = TargetOverlayRenderer.render(resources, shotEditor.snapshot(), selectedIndex)
        maxRadius = copy.width / 2f
        imageView.setImageBitmap(copy)
    }

    private fun handleTouchDown(x: Float, y: Float) {
        val point = eventToShotPoint(x, y)
        if (addMode) {
            shotEditor.add(point)
            selectedIndex = shotEditor.snapshot().lastIndex
            addMode = false
            btnAdd.text = "Добавить"
            redraw()
            return
        }

        val nearestIndex = findNearestPoint(point)
        if (nearestIndex != -1) {
            selectedIndex = nearestIndex
            dragMode = true
            redraw()
        }
    }

    private fun handleTouchMove(x: Float, y: Float) {
        if (!dragMode || selectedIndex !in shotEditor.snapshot().indices) return
        shotEditor.move(selectedIndex, eventToShotPoint(x, y))
        redraw()
    }

    private fun eventToShotPoint(x: Float, y: Float): ShotPoint {
        val cx = imageView.width / 2f
        val cy = imageView.height / 2f
        val maxR = minOf(cx, cy)
        return ShotPoint(
            ((x - cx) / maxR).coerceIn(-1f, 1f),
            ((y - cy) / maxR).coerceIn(-1f, 1f)
        )
    }

    private fun findNearestPoint(point: ShotPoint): Int {
        val points = shotEditor.snapshot()
        var nearestIndex = -1
        var nearestDistance = Float.MAX_VALUE
        for ((index, candidate) in points.withIndex()) {
            val dx = candidate.xNorm - point.xNorm
            val dy = candidate.yNorm - point.yNorm
            val distance = sqrt(dx * dx + dy * dy)
            if (distance < nearestDistance) {
                nearestDistance = distance
                nearestIndex = index
            }
        }
        return if (nearestDistance <= 0.12f) nearestIndex else -1
    }

    private fun normalizeSelectedIndex() {
        val points = shotEditor.snapshot()
        selectedIndex = when {
            points.isEmpty() -> -1
            selectedIndex !in points.indices -> points.lastIndex
            else -> selectedIndex
        }
    }

    private fun polarToShotPoint(radius: Float, angleDeg: Float): ShotPoint {
        val angleRad = angleDeg / 180f * PI
        return ShotPoint(
            (radius * cos(angleRad)).toFloat(),
            (radius * sin(angleRad)).toFloat()
        )
    }

    private fun angleDeg(point: ShotPoint): Float {
        return Math.toDegrees(atan2(point.yNorm.toDouble(), point.xNorm.toDouble())).toFloat()
    }

    companion object {
        private const val STATE_X_POINTS = "state_x_points"
        private const val STATE_Y_POINTS = "state_y_points"
        private const val STATE_SELECTED_INDEX = "state_selected_index"
        private const val STATE_EDIT_MODE = "state_edit_mode"
        private const val STATE_ADD_MODE = "state_add_mode"
    }
}
