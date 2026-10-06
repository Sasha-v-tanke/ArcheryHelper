package com.direwolf.archeryhelper.stats

import com.direwolf.archeryhelper.domain.ShotPoint
import com.direwolf.archeryhelper.domain.TargetConfig
import com.direwolf.archeryhelper.domain.TargetFormat
import com.direwolf.archeryhelper.domain.TargetTemplate
import com.direwolf.archeryhelper.domain.TenRingMode
import kotlin.math.max
import kotlin.math.sqrt

data class ScoreResult(
    val finalCandidate: Int,
    val isX: Boolean,
    val lineCall: Boolean
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
    fun score(
        targetTemplate: TargetTemplate,
        impact: ShotPoint,
        arrowDiameterMm: Float? = null,
        localizationError: Float = 0f
    ): ScoreResult

    fun analyze(targetTemplate: TargetTemplate, shots: List<ShotPoint>): SeriesStatistics
}

class ArcheryScoringEngine : ScoringEngine {
    override fun score(
        targetTemplate: TargetTemplate,
        impact: ShotPoint,
        arrowDiameterMm: Float?,
        localizationError: Float
    ): ScoreResult {
        require(localizationError.isFinite() && localizationError >= 0f)

        val diameter = arrowDiameterMm ?: targetTemplate.config.arrowDiameterMm
        require(diameter == null || (diameter.isFinite() && diameter > 0f))

        val shaftRadiusNorm = if (diameter == null) {
            0f
        } else {
            diameter / targetTemplate.config.faceDiameterMm
        }
        val centerRadius = impact.radiusNorm()
        val scoringRadius = max(0f, centerRadius - shaftRadiusNorm)
        val finalCandidate = targetTemplate.ringBoundaries
            .firstOrNull { scoringRadius <= it.radiusNorm }
            ?.score
            ?: 0
        val isX = finalCandidate == 10 && scoringRadius <= targetTemplate.xRingRadiusNorm
        val lineCall = isLineCall(
            targetTemplate = targetTemplate,
            centerRadius = centerRadius,
            shaftRadiusNorm = shaftRadiusNorm,
            localizationError = localizationError
        )
        return ScoreResult(finalCandidate, isX, lineCall)
    }

    override fun analyze(targetTemplate: TargetTemplate, shots: List<ShotPoint>): SeriesStatistics {
        if (shots.isEmpty()) {
            return SeriesStatistics(0, 0.0, ShotPoint(0f, 0f), 0f, 0f, 0f, 0f, emptyMap())
        }

        val scores = shots.map {
            val result = score(targetTemplate, it)
            if (result.isX) 11 else result.finalCandidate
        }
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

    private fun isLineCall(
        targetTemplate: TargetTemplate,
        centerRadius: Float,
        shaftRadiusNorm: Float,
        localizationError: Float
    ): Boolean {
        if (localizationError == 0f) return false

        val lower = max(0f, centerRadius - localizationError - shaftRadiusNorm)
        val upper = max(0f, centerRadius + localizationError - shaftRadiusNorm)
        val boundaries = targetTemplate.ringBoundaries.map { it.radiusNorm } + targetTemplate.xRingRadiusNorm
        return boundaries.any { it in lower..upper }
    }
}

object ScoreCalculator {
    private val defaultEngine: ScoringEngine = ArcheryScoringEngine()
    private val defaultTarget = TargetTemplate.from(
        TargetConfig(
            format = TargetFormat.SINGLE,
            tripleLayout = null,
            minimumScoringZone = 1,
            tenRingMode = TenRingMode.RECURVE,
            faceDiameterMm = 400
        )
    )

    fun scoreRadius(radiusNorm: Float): Int {
        val result = defaultEngine.score(defaultTarget, ShotPoint(radiusNorm, 0f))
        return if (result.isX) 11 else result.finalCandidate
    }
}
