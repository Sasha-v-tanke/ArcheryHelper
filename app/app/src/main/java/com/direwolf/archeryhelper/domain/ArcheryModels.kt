package com.direwolf.archeryhelper.domain

import kotlin.math.PI
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

enum class InputMode {
    MANUAL,
    SCAN
}

enum class TargetFormat {
    SINGLE,
    TRIPLE
}

enum class TripleLayout {
    VERTICAL,
    TRIANGULAR
}

enum class TenRingMode {
    RECURVE,
    COMPOUND
}

enum class ShotSource {
    MODEL,
    MANUAL,
    CORRECTED_MODEL
}

data class TargetConfig(
    val format: TargetFormat,
    val tripleLayout: TripleLayout?,
    val minimumScoringZone: Int,
    val tenRingMode: TenRingMode,
    val faceDiameterMm: Int,
    val arrowDiameterMm: Float? = null,
    val expectedArrowsPerSeries: Int? = null
) {
    init {
        require(minimumScoringZone in setOf(1, 5, 6))
        require((format == TargetFormat.SINGLE && tripleLayout == null) || (format == TargetFormat.TRIPLE && tripleLayout != null))
        require(faceDiameterMm > 0)
        require(arrowDiameterMm == null || arrowDiameterMm > 0f)
        require(expectedArrowsPerSeries == null || expectedArrowsPerSeries > 0)
    }
}

data class RingBoundary(
    val score: Int,
    val radiusNorm: Float
)

data class TargetTemplate(
    val config: TargetConfig,
    val ringBoundaries: List<RingBoundary>,
    val xRingRadiusNorm: Float
) {
    val visibleRadiusNorm: Float
        get() = ringBoundaries.last().radiusNorm

    companion object {
        fun from(config: TargetConfig): TargetTemplate {
            val boundaries = (10 downTo config.minimumScoringZone).map { score ->
                val radius = if (score == 10 && config.tenRingMode == TenRingMode.COMPOUND) {
                    0.05f
                } else {
                    (11 - score) * 0.1f
                }
                RingBoundary(score, radius)
            }
            return TargetTemplate(
                config = config,
                ringBoundaries = boundaries,
                xRingRadiusNorm = 0.05f
            )
        }
    }
}

data class TrainingSession(
    val date: String,
    val number: Int,
    val distance: Int,
    val series: MutableList<Series> = mutableListOf(),
    val inputMode: InputMode = InputMode.MANUAL,
    val createdAtMillis: Long = 0L,
    val targetConfig: TargetConfig? = null
)

typealias Distance = TrainingSession

data class Series(
    val number: Int,
    val shots: MutableList<Shot> = mutableListOf()
)

data class Shot(
    var number: Int,
    val result: Int,
    val xNorm: Float? = null,
    val yNorm: Float? = null,
    val faceIndex: Int = 0,
    val confidence: Float? = null,
    val source: ShotSource = ShotSource.MANUAL,
    val corrected: Boolean = false,
    val lineCall: Boolean = false,
    val modelVersion: String? = null
) {
    init {
        require((xNorm == null) == (yNorm == null))
        require(faceIndex >= 0)
        require(confidence == null || confidence in 0f..1f)
    }

    val normalizedScore: Int
        get() = if (result == 11) 10 else result

    val radiusNorm: Float?
        get() = if (xNorm == null || yNorm == null) null else sqrt(xNorm * xNorm + yNorm * yNorm)

    val angleDeg: Float?
        get() = if (xNorm == null || yNorm == null) null else Math.toDegrees(atan2(yNorm.toDouble(), xNorm.toDouble())).toFloat()

    val distance: Float?
        get() = radiusNorm

    val angle: Float?
        get() = angleDeg
}

data class ShotPoint(
    val xNorm: Float,
    val yNorm: Float,
    val faceIndex: Int = 0
) {
    init {
        require(xNorm.isFinite() && yNorm.isFinite())
        require(faceIndex >= 0)
    }

    fun radiusNorm(): Float = sqrt(xNorm * xNorm + yNorm * yNorm)

    fun angleDeg(): Float = Math.toDegrees(atan2(yNorm.toDouble(), xNorm.toDouble())).toFloat()

    companion object {
        fun fromPolar(radiusNorm: Float, angleDeg: Float, faceIndex: Int = 0): ShotPoint {
            require(radiusNorm.isFinite() && angleDeg.isFinite())
            val angleRad = angleDeg / 180f * PI
            return ShotPoint(
                (radiusNorm * cos(angleRad)).toFloat(),
                (radiusNorm * sin(angleRad)).toFloat(),
                faceIndex
            )
        }
    }
}

data class DetectedImpact(
    val confidence: Float,
    val xNorm: Float,
    val yNorm: Float,
    val faceIndex: Int = 0
)

typealias DetectedShot = DetectedImpact
