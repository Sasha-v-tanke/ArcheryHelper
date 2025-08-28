package com.direwolf.archeryhelper.widgets

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.direwolf.archeryhelper.R
import com.direwolf.archeryhelper.utils.Distance
import com.direwolf.archeryhelper.utils.Series

class DistancesListAdapter(
    private val distances: List<Distance>,
    private val onClick: (Int) -> Unit
) : RecyclerView.Adapter<DistancesListAdapter.DistanceViewHolder>() {

    class DistanceViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        val tvIndex: TextView = itemView.findViewById(R.id.tvDistanceIndex)
        val tvDate: TextView = itemView.findViewById(R.id.tvDistanceDate)
        val tvDistance: TextView = itemView.findViewById(R.id.tvDistance)
        val tvSummary: TextView = itemView.findViewById(R.id.tvSum)
        val tvAverage: TextView = itemView.findViewById(R.id.tvAvg)
        val tvSeriesCount: TextView = itemView.findViewById(R.id.tvSeriesCount)
        val tvShotsCount: TextView = itemView.findViewById(R.id.tvShotsCount)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): DistanceViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_distance, parent, false)
        return DistanceViewHolder(view)
    }

    override fun onBindViewHolder(holder: DistanceViewHolder, position: Int) {
        val distance = distances[position]

        holder.tvIndex.text = "#${position + 1}"
        holder.tvDate.text = distance.date
        holder.tvDistance.text = "${distance.distance}м"

        val seriesCount = distance.series.size
        val shotsCount = distance.series.sumOf { it.shots.size }
        val sum = distance.series.sumOf { s -> s.shots.sumOf { it.result } }
        val avg = if (shotsCount > 0) sum.toDouble() / shotsCount else 0.0

        holder.tvSummary.text = "Сумма: $sum"
        holder.tvAverage.text = "Среднее: " + String.format("%.1f", avg)
        holder.tvSeriesCount.text = "Серий: $seriesCount"
        holder.tvShotsCount.text = "Выстрелов: $shotsCount"

        holder.itemView.setOnClickListener {
            onClick(position + 1)
        }
    }

    override fun getItemCount(): Int = distances.size
}