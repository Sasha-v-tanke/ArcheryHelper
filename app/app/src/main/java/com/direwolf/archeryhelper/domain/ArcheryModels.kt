package com.direwolf.archeryhelper.domain

import kotlin.math.sqrt

enum class InputMode {
    MANUAL,
    SCAN
}

data class Distance(
    val date: String,
    val number: Int,
    val distance: Int,
    val series: MutableList<Series> = mutableListOf(),
    val inputMode: InputMode = InputMode.MANUAL,
    val createdAtMillis: Long = 0L
)

data class Series(
    val number: Int,
    val shots: MutableList<Shot> = mutableListOf()
)

data class Shot(
    var number: Int,
    val result: Int,
    val distance: Float? = null,
    val angle: Float? = null
) {
    val normalizedScore: Int
        get() = if (result == 11) 10 else result
}

data class ShotPoint(
    val xNorm: Float,
    val yNorm: Float
) {
    fun radiusNorm(): Float = sqrt(xNorm * xNorm + yNorm * yNorm)
}

data class DetectedShot(
    val confidence: Float,
    val xNorm: Float,
    val yNorm: Float
)
