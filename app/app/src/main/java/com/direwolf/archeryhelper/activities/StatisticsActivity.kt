package com.direwolf.archeryhelper.activities

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.direwolf.archeryhelper.R
import com.direwolf.archeryhelper.managers.DataManager
import com.direwolf.archeryhelper.utils.Distance
import com.direwolf.archeryhelper.widgets.DistancesListAdapter

class StatisticsActivity : TemplateActivity() {
    override fun getLayoutId(): Int = R.layout.activity_statistics

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val backBtn: Button = findViewById(R.id.btnBack)
        backBtn.setOnClickListener {
            startActivity(Intent(this, MainMenuActivity::class.java))
            finish()
        }

        val recyclerView: RecyclerView = findViewById(R.id.distancesList)
        recyclerView.layoutManager = LinearLayoutManager(this)

        val distances = mutableListOf<Distance>()
        for (i in 1..DataManager.getLastDistanceIndex()) {
            distances.add(DataManager.loadDistance(i))
        }

        if (distances.size == 0) {
            findViewById<TextView>(R.id.empty).visibility = TextView.VISIBLE
            recyclerView.visibility = RecyclerView.GONE
        } else {
            findViewById<TextView>(R.id.empty).visibility = TextView.GONE
            recyclerView.visibility = RecyclerView.VISIBLE
            recyclerView.adapter = DistancesListAdapter(distances) { index ->
                val intent = Intent(this, DetailedStatisticsActivity::class.java)
                intent.putExtra("distance_index", index)
                startActivity(intent)
            }
        }
    }
}