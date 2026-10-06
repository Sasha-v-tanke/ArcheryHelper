package com.direwolf.archeryhelper.activities

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import com.direwolf.archeryhelper.R
import com.direwolf.archeryhelper.managers.DataManager

class NewDistanceActivity : TemplateActivity() {

    companion object {
        const val EXTRA_INPUT_TYPE = "input_type"
        const val INPUT_MANUAL = "manual"
        const val INPUT_SCAN = "scan"
    }

    override fun getLayoutId(): Int = R.layout.activity_new_distance

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        findViewById<Button>(R.id.btnManualInput).setOnClickListener {
            openDistanceTable(INPUT_MANUAL)
        }

        findViewById<Button>(R.id.btnScanInput).setOnClickListener {
            openDistanceTable(INPUT_SCAN)
        }

        findViewById<Button>(R.id.btnBack).setOnClickListener {
            startActivity(Intent(this, MainMenuActivity::class.java))
            finish()
        }
    }

    private fun openDistanceTable(inputType: String) {
        DataManager.startNewDistance(inputType == INPUT_MANUAL)
        startActivity(Intent(this, DistanceActivity::class.java))
        finish()
    }
}