package com.direwolf.archeryhelper.activities

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import com.direwolf.archeryhelper.R
import com.direwolf.archeryhelper.managers.DataManager
import com.direwolf.archeryhelper.utils.Series
import com.direwolf.archeryhelper.utils.Shot
import com.direwolf.archeryhelper.utils.debugLog

class MainMenuActivity : TemplateActivity() {
    override fun getLayoutId(): Int = R.layout.activity_main_menu

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        findViewById<Button>(R.id.btnNewDistance).setOnClickListener {
            startActivity(Intent(this, NewDistanceActivity::class.java))
        }

        val btnContinueDistance = findViewById<Button>(R.id.btnContinueDistance)
        btnContinueDistance.setOnClickListener {
            startActivity(Intent(this, DistanceActivity::class.java))
        }

        if (DataManager.getLastDistanceIndex() == 0) {
            btnContinueDistance.isEnabled = false
        }

        findViewById<Button>(R.id.btnShowDistances).setOnClickListener {
            startActivity(Intent(this, StatisticsActivity::class.java))
        }
//        debug()

    }

    private fun debug() {
        DataManager.clearAllData()
        DataManager.dumpPrefs()
        val lst1 = listOf(9, 4, 5, 6, 3)
        val lst2 = listOf(5, 5, 7, 9)
        val series1 = Series(1)
        val series2 = Series(2)
        for (i in lst1.indices) {
            series1.shots.add(Shot(i + 1, lst1[i]))
        }
        for (i in lst2.indices) {
            series2.shots.add(Shot(i + 1, lst2[i]))
        }
        DataManager.startNewDistance(true)
        DataManager.saveSeries(series1, 1)
        DataManager.saveSeries(series2, 1)
    }
}