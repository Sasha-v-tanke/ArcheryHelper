package com.direwolf.archeryhelper.activities

import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import com.direwolf.archeryhelper.R
import com.direwolf.archeryhelper.domain.ShotPoint
import com.direwolf.archeryhelper.managers.DataManager
import com.direwolf.archeryhelper.ui.TargetOverlayRenderer
import com.direwolf.archeryhelper.utils.Shot

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
        val points = shots.mapNotNull { shot ->
            if (shot.xNorm == null || shot.yNorm == null) return@mapNotNull null
            ShotPoint(shot.xNorm, shot.yNorm, shot.faceIndex)
        }
        imageView.setImageBitmap(TargetOverlayRenderer.render(resources, points, pointRadius = 4f))
    }
}
