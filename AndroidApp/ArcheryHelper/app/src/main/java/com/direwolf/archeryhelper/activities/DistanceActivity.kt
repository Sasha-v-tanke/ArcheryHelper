package com.direwolf.archeryhelper.activities

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.recyclerview.widget.GridLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.direwolf.archeryhelper.R
import com.direwolf.archeryhelper.managers.DataManager
import com.direwolf.archeryhelper.utils.Distance
import com.direwolf.archeryhelper.utils.Series
import com.direwolf.archeryhelper.utils.debugLog
import com.direwolf.archeryhelper.widgets.SeriesListAdapter

class DistanceActivity : TemplateActivity() {

    private lateinit var recyclerView: RecyclerView
    private lateinit var distance: Distance

    override fun getLayoutId(): Int = R.layout.activity_distance

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        recyclerView = findViewById(R.id.seriesListView)

        distance = DataManager.loadLastDistance()
        recyclerView.layoutManager = GridLayoutManager(this, 5, RecyclerView.HORIZONTAL, false)
        recyclerView.adapter = SeriesListAdapter(distance)

        var count = 0
        var sum = 0
        for (series in distance.series) {
            count += series.shots.size
            for (shot in series.shots) {
                sum += shot.result
            }
        }

        val avg = if (count != 0) sum.toDouble() / count else 0.0
        findViewById<TextView>(R.id.seriesAvg).text = "Average: " + String.format("%.1f", avg)
        findViewById<TextView>(R.id.seriesSum).text = "Sum: $sum"
        findViewById<TextView>(R.id.seriesCnt).text = "Count: $count"

        findViewById<Button>(R.id.btnAddSeries).setOnClickListener {
            if (DataManager.isManualInput(distance.number)) {
                startActivity(Intent(this, ManualInputActivity::class.java))
                finish()
            } else {
                startActivity(Intent(this, ScanActivity::class.java))
            }

        }

        findViewById<Button>(R.id.btnCurrentStatistics).setOnClickListener {
            val intent = Intent(this, DetailedStatisticsActivity::class.java)
            intent.putExtra("distance_index", DataManager.getLastDistanceIndex())
            startActivity(intent)
        }

        findViewById<Button>(R.id.btnSaveDistance).setOnClickListener {
            startActivity(Intent(this, MainMenuActivity::class.java))
            finish()
        }
    }
}