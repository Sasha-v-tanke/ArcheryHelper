package com.direwolf.archeryhelper.widgets

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.direwolf.archeryhelper.R
import com.direwolf.archeryhelper.utils.Series
import com.direwolf.archeryhelper.utils.Shot

class ShotListAdapter(
    private val series: Series
) : RecyclerView.Adapter<ShotListAdapter.ShotViewHolder>() {

    class ShotViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        val tvShotNumber: TextView = itemView.findViewById(R.id.tvShotNumber)
        val tvShotResult: TextView = itemView.findViewById(R.id.tvShotResult)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ShotViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_shot, parent, false)
        return ShotViewHolder(view)
    }

    override fun onBindViewHolder(holder: ShotViewHolder, position: Int) {
        val shot: Shot = series.shots[position]
        holder.tvShotNumber.text = "Выстрел ${shot.number}"
        holder.tvShotResult.text = shot.result.toString()
    }

    override fun getItemCount(): Int = series.shots.size
}