package com.direwolf.archeryhelper.activities

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import com.direwolf.archeryhelper.R
import com.direwolf.archeryhelper.managers.DataManager
import com.direwolf.archeryhelper.utils.Series
import com.direwolf.archeryhelper.utils.Shot
//import com.direwolf.archeryhelper.utils.convert
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

        if (DataManager.getLastDistanceIndex() == 0)
            btnContinueDistance.visibility = Button.GONE
        else
            btnContinueDistance.visibility = Button.VISIBLE

        findViewById<Button>(R.id.btnShowDistances).setOnClickListener {
            startActivity(Intent(this, StatisticsActivity::class.java))
        }
//        debug()

        findViewById<Button>(R.id.btnClear).setOnClickListener {
//            DataManager.clearAllData()
            debug()
        }

    }

    private fun debug() {
//        convert(this)
    }
}