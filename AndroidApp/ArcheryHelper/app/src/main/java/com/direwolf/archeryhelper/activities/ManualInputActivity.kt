package com.direwolf.archeryhelper.activities

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.NumberPicker
import android.widget.TextView
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.direwolf.archeryhelper.R
import com.direwolf.archeryhelper.managers.DataManager
import com.direwolf.archeryhelper.utils.Series
import com.direwolf.archeryhelper.utils.Shot
import com.direwolf.archeryhelper.utils.debugLog
import com.direwolf.archeryhelper.widgets.ShotListAdapter

class ManualInputActivity : TemplateActivity() {

    private lateinit var recyclerView: RecyclerView
    private lateinit var adapter: ShotListAdapter
    private lateinit var series: Series
    private lateinit var sumView: TextView
    private lateinit var avgView: TextView

    override fun getLayoutId(): Int = R.layout.activity_manual_input

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        recyclerView = findViewById(R.id.recyclerView)
        series = Series(DataManager.getLastSeriesIndex() + 1)
        adapter = ShotListAdapter(series)
        recyclerView.layoutManager = LinearLayoutManager(this)
        recyclerView.adapter = adapter

        findViewById<Button>(R.id.btnSave).setOnClickListener {
            series.shots.sortByDescending { it.result }
            series.shots.forEachIndexed { index, shot -> shot.number = index + 1 }
            DataManager.saveSeries(series)
            startActivity(Intent(this, DistanceActivity::class.java))
        }

        findViewById<Button>(R.id.btnRemove).setOnClickListener {
            series.shots.removeAt(series.shots.size - 1)
            adapter.notifyDataSetChanged()
        }

        findViewById<Button>(R.id.btnAddShot).setOnClickListener {
            showNumberPicker()
        }

        sumView = findViewById(R.id.shotsSum)
        avgView = findViewById(R.id.shotsAvg)
    }

    private fun showNumberPicker() {
        val numberPicker = NumberPicker(this)
        numberPicker.minValue = 0
        numberPicker.maxValue = 10
        numberPicker.wrapSelectorWheel = false

        val dialog = android.app.AlertDialog.Builder(this)
            .setTitle("Выберите результат выстрела")
            .setView(numberPicker)
            .setPositiveButton("OK") { _, _ ->
                val selectedResult = numberPicker.value
                addShot(selectedResult)
            }
            .setNegativeButton("Отмена", null)
            .create()

        dialog.show()
    }

    private fun addShot(result: Int) {
        series.shots.add(Shot(series.shots.size + 1, result))
        adapter.notifyDataSetChanged()
    }
}