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
            finish()
        }

        val distanceIndex = intent.getIntExtra("distance_index", 1)
        distance = DataManager.loadDistance(distanceIndex)
        findViewById<TextView>(R.id.description).text = "${distance.date}\nДистанция: ${distance.distance}м"

        recyclerView = findViewById(R.id.recyclerSeries)
        recyclerView.layoutManager = LinearLayoutManager(this)
        recyclerView.adapter = SeriesListAdapter(distance)

        val shotsCount = distance.series.sumOf { it.shots.size }
        val sum = distance.series.sumOf { s -> s.shots.sumOf { if (it.result == 11) 10 else it.result } }
        val avg = if (shotsCount > 0) sum.toDouble() / shotsCount else 0.0

        if (shotsCount == 0) {
            findViewById<TextView>(R.id.empty).visibility = TextView.VISIBLE
            recyclerView.visibility = RecyclerView.GONE
        } else {
            findViewById<TextView>(R.id.empty).visibility = TextView.GONE
            recyclerView.visibility = RecyclerView.VISIBLE
        }

        findViewById<TextView>(R.id.distanceSum).text = "Сумма: $sum"
        findViewById<TextView>(R.id.shotsAvg).text = "Среднее: ${String.format("%.1f", avg)}"
        findViewById<TextView>(R.id.shotsCount).text = "Выстрелы: $shotsCount"

        val showBtn = findViewById<Button>(R.id.showAnalyses)
        if (!DataManager.isManualInput(distance.number)) showBtn.visibility = Button.VISIBLE
        else showBtn.visibility = Button.GONE
        showBtn.setOnClickListener {
            val intent = Intent(this, AdvancedStatisticsActivity::class.java)
            intent.putExtra("index", distanceIndex)
            startActivity(intent)
        }


    }
}