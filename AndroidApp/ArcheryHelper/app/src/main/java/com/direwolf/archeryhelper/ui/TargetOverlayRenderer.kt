package com.direwolf.archeryhelper.ui

import android.content.res.Resources
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import com.direwolf.archeryhelper.R
import com.direwolf.archeryhelper.domain.ShotPoint

object TargetOverlayRenderer {
    fun render(
        resources: Resources,
        points: List<ShotPoint>,
        selectedIndex: Int = -1,
        pointRadius: Float = 12f
    ): Bitmap {
        val bitmap = BitmapFactory.decodeResource(resources, R.drawable.target)
        val scaled = Bitmap.createScaledBitmap(bitmap, 800, 800, true)
        val copy = scaled.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(copy)

        val paintNormal = Paint().apply {
            color = Color.GREEN
            style = Paint.Style.FILL
            strokeWidth = 10f
        }
        val paintSelected = Paint().apply {
            color = Color.CYAN
            style = Paint.Style.FILL
            strokeWidth = 12f
        }

        val maxRadius = copy.width / 2f
        val cx = maxRadius
        val cy = maxRadius
        for ((i, point) in points.withIndex()) {
            val x = cx + point.xNorm * maxRadius
            val y = cy + point.yNorm * maxRadius
            val paint = if (i == selectedIndex) paintSelected else paintNormal
            canvas.drawCircle(x, y, pointRadius, paint)
        }

        return copy
    }
}
