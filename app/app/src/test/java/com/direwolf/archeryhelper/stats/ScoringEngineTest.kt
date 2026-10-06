package com.direwolf.archeryhelper.stats

import com.direwolf.archeryhelper.domain.ShotPoint
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ScoringEngineTest {
    private val engine = ArcheryScoringEngine()
    private val target = TargetGeometry()

    @Test
    fun scoreReturnsXForCenterShot() {
        assertEquals(11, engine.score(target, ShotPoint(0f, 0f)))
    }

    @Test
    fun scoreReturnsTenForInnerGoldOutsideX() {
        assertEquals(10, engine.score(target, ShotPoint(0.06f, 0f)))
    }

    @Test
    fun scoreReturnsZeroOutsideTarget() {
        assertEquals(0, engine.score(target, ShotPoint(1.2f, 0f)))
    }

    @Test
    fun analyzeCalculatesSeriesStatistics() {
        val stats = engine.analyze(
            target,
            listOf(
                ShotPoint(0f, 0f),
                ShotPoint(0.25f, 0f),
                ShotPoint(-0.25f, 0f)
            )
        )

        assertEquals(26, stats.totalScore)
        assertEquals(26.0 / 3.0, stats.averageScore, 0.0001)
        assertEquals(0f, stats.groupCenter.xNorm, 0.0001f)
        assertEquals(0f, stats.groupCenter.yNorm, 0.0001f)
        assertTrue(stats.maxSpread > 0.49f)
        assertEquals(1, stats.ringCounts[11])
        assertEquals(2, stats.ringCounts[8])
    }

    @Test
    fun emptySeriesHasZeroStatistics() {
        val stats = engine.analyze(target, emptyList())

        assertEquals(0, stats.totalScore)
        assertEquals(0.0, stats.averageScore, 0.0)
        assertEquals(emptyMap<Int, Int>(), stats.ringCounts)
    }
}
