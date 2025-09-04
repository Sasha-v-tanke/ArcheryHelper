package com.direwolf.archeryhelper.activities

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.Bundle
import android.os.PersistableBundle
import android.provider.ContactsContract.Data
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.direwolf.archeryhelper.R
import com.direwolf.archeryhelper.managers.Application
import com.direwolf.archeryhelper.managers.DataManager
import com.direwolf.archeryhelper.utils.Distance
import com.direwolf.archeryhelper.utils.Shot
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

class AdvancedStatisticsActivity : TemplateActivity() {
    override fun getLayoutId(): Int = R.layout.activity_advanced_statistics
    private val shots = mutableListOf<Shot>()
    private lateinit var imageView: ImageView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        var index = intent.getIntExtra("index", -1)
        if (index == -1) index = DataManager.getLastDistanceIndex()
        val distance = DataManager.loadDistance(index)
        for (series in distance.series) {
            for (shot in series.shots) {
                shots.add(shot)
            }
        }
        findViewById<Button>(R.id.btnBack).setOnClickListener {
            finish()
        }
        imageView = findViewById(R.id.imageView)
        imageView.post {
            redraw()
        }
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
            color = Color.CYAN
            style = Paint.Style.FILL
            strokeWidth = 12f
        }

        val maxRadius = copy.width / 2f
        val cx = maxRadius
        val cy = maxRadius
        for (shot in shots) {
            if (shot.distance == null || shot.angle == null) continue
            val r = shot.distance
            val theta = shot.angle
            val x = cx + r * cos(theta / 180f * PI) * maxRadius
            val y = cy + r * sin(theta / 180f * PI) * maxRadius
            canvas.drawCircle(x.toFloat(), y.toFloat(), 4f, paintNormal)
        }

        imageView.setImageBitmap(copy)
    }
}