package com.direwolf.archeryhelper.stats

import com.direwolf.archeryhelper.domain.ShotPoint
import kotlin.math.max
import kotlin.math.sqrt

data class TargetGeometry(
    val rings: Int = 10,
    val xRingRadiusNorm: Float = 0.05f
)

data class SeriesStatistics(
    val totalScore: Int,
    val averageScore: Double,
    val groupCenter: ShotPoint,
    val meanRadialDistance: Float,
    val maxSpread: Float,
    val horizontalBias: Float,
    val verticalBias: Float,
    val ringCounts: Map<Int, Int>
)

interface ScoringEngine {
    fun score(target: TargetGeometry, shot: ShotPoint): Int
    fun analyze(target: TargetGeometry, shots: List<ShotPoint>): SeriesStatistics
}

class ArcheryScoringEngine : ScoringEngine {
    override fun score(target: TargetGeometry, shot: ShotPoint): Int {
        val radius = shot.radiusNorm()
        if (radius <= target.xRingRadiusNorm) return 11
        if (radius > 1f) return 0
        return max(0, target.rings - (radius * target.rings).toInt())
    }

    override fun analyze(target: TargetGeometry, shots: List<ShotPoint>): SeriesStatistics {
        if (shots.isEmpty()) {
            return SeriesStatistics(0, 0.0, ShotPoint(0f, 0f), 0f, 0f, 0f, 0f, emptyMap())
        }

        val scores = shots.map { score(target, it) }
        val center = ShotPoint(
            shots.sumOf { it.xNorm.toDouble() }.toFloat() / shots.size,
            shots.sumOf { it.yNorm.toDouble() }.toFloat() / shots.size
        )
        val distancesFromCenter = shots.map {
            val dx = it.xNorm - center.xNorm
            val dy = it.yNorm - center.yNorm
            sqrt(dx * dx + dy * dy)
        }
        val maxSpread = shots.flatMapIndexed { i, a ->
            shots.drop(i + 1).map { b ->
                val dx = a.xNorm - b.xNorm
                val dy = a.yNorm - b.yNorm
                sqrt(dx * dx + dy * dy)
            }
        }.maxOrNull() ?: 0f

        return SeriesStatistics(
            totalScore = scores.sumOf { if (it == 11) 10 else it },
            averageScore = scores.sumOf { if (it == 11) 10 else it }.toDouble() / shots.size,
            groupCenter = center,
            meanRadialDistance = distancesFromCenter.sum() / distancesFromCenter.size,
            maxSpread = maxSpread,
            horizontalBias = center.xNorm,
            verticalBias = center.yNorm,
            ringCounts = scores.groupingBy { it }.eachCount()
        )
    }
}

object ScoreCalculator {
    private val defaultEngine = ArcheryScoringEngine()
    private val defaultTarget = TargetGeometry()

    fun scoreRadius(radiusNorm: Float): Int {
        return defaultEngine.score(defaultTarget, ShotPoint(radiusNorm, 0f))
    }
}
