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
            series.shots.sortBy { -it.result }
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
        val labels = (0..10).map { it.toString() }.plus("X").toTypedArray()
        val scores = ((0..10).toMutableList().apply { add(11) }).asReversed()

        val numberPicker = NumberPicker(this).apply {
            wrapSelectorWheel = false
            descendantFocusability = NumberPicker.FOCUS_BLOCK_DESCENDANTS
            minValue = 0
            maxValue = labels.size - 1
            displayedValues = labels.reversedArray()
        }

        val dialog = android.app.AlertDialog.Builder(this)
            .setTitle("Выберите результат выстрела")
            .setView(numberPicker)
            .setPositiveButton("OK") { _, _ ->
                val selectedResult = scores[numberPicker.value] // 11 для "X"
                addShot(selectedResult)
                update()
            }
            .setNegativeButton("Отмена", null)
            .create()

        dialog.show()
    }

    private fun addShot(result: Int) {
        series.shots.add(Shot(series.shots.size + 1, result))
        adapter.notifyDataSetChanged()
    }

    private fun update() {
        val sum = series.shots.sumOf { if (it.result == 11) 10 else it.result }
        val avg = if (series.shots.size > 0) sum.toDouble() / series.shots.size else 0.0

        sumView.text = "Сумма: $sum"
        avgView.text = "Среднее: ${String.format("%.1f", avg)}"
    }
}