package com.direwolf.archeryhelper.widgets

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.direwolf.archeryhelper.R
import com.direwolf.archeryhelper.utils.Distance
import com.direwolf.archeryhelper.utils.Series

class SeriesListAdapter(
    private val distance: Distance
) : RecyclerView.Adapter<SeriesListAdapter.SeriesViewHolder>() {

    class SeriesViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        val tvSeriesNumber: TextView = itemView.findViewById(R.id.tvSeriesNumber)
        val llShotsContainer: LinearLayout = itemView.findViewById(R.id.shotsContainer)
        val tvSeriesSum: TextView = itemView.findViewById(R.id.tvSeriesSum)
        val tvSeriesAvg: TextView = itemView.findViewById(R.id.tvSeriesAvg)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): SeriesViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_series, parent, false)
        return SeriesViewHolder(view)
    }

    override fun onBindViewHolder(holder: SeriesViewHolder, position: Int) {
        val series = distance.series[position]
        holder.tvSeriesNumber.text = "Серия ${series.number}"

        holder.llShotsContainer.removeAllViews()
        series.shots.forEach { shot ->
            val shotTextView = TextView(holder.itemView.context)
            shotTextView.text = if (shot.result == 11) "X" else shot.result.toString()
            shotTextView.textSize = 16f
            shotTextView.setPadding(8, 0, 8, 0)
            holder.llShotsContainer.addView(shotTextView)
        }

        val sum = series.shots.sumOf { if (it.result == 1) 10 else it.result }
        holder.tvSeriesSum.text = sum.toString()

        val avg = if (series.shots.isNotEmpty()) sum.toDouble() / series.shots.size else 0.0
        holder.tvSeriesAvg.text = String.format("%.1f", avg)
    }

    override fun getItemCount(): Int = distance.series.size
}