package com.direwolf.archeryhelper.activities

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.ImageButton
import android.widget.TextView
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.direwolf.archeryhelper.R
import com.direwolf.archeryhelper.managers.DataManager
import com.direwolf.archeryhelper.utils.Distance
import com.direwolf.archeryhelper.widgets.SeriesListAdapter

class DetailedStatisticsActivity : TemplateActivity() {

    private lateinit var recyclerView: RecyclerView
    private lateinit var distance: Distance

    override fun getLayoutId(): Int = R.layout.activity_detailed_statistics

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val backBtn: Button = findViewById(R.id.btnBack)
        backBtn.setOnClickListener {
            startActivity(Intent(this, StatisticsActivity::class.java))
            finish()
        }

        val distanceIndex = intent.getIntExtra("distance_index", 1)
        distance = DataManager.loadDistance(distanceIndex)
        findViewById<TextView>(R.id.description).text = "${distance.date}\nДистанция: ${distance.distance}м"

        recyclerView = findViewById(R.id.recyclerSeries)
        recyclerView.layoutManager = LinearLayoutManager(this)
        recyclerView.adapter = SeriesListAdapter(distance)

        val shotsCount = distance.series.sumOf { it.shots.size }
        val sum = distance.series.sumOf { s -> s.shots.sumOf { it.result } }
        val avg = if (shotsCount > 0) sum.toDouble() / shotsCount else 0.0

        findViewById<TextView>(R.id.distanceSum).text = "Сумма: $sum"
        findViewById<TextView>(R.id.shotsAvg).text = "Среднее: ${String.format("%.1f", avg)}"
        findViewById<TextView>(R.id.shotsCount).text = "Выстрелов: $shotsCount"

    }
}