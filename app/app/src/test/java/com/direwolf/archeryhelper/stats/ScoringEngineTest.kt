package com.direwolf.archeryhelper.stats

import com.direwolf.archeryhelper.domain.ShotPoint
import com.direwolf.archeryhelper.domain.TargetConfig
import com.direwolf.archeryhelper.domain.TargetFormat
import com.direwolf.archeryhelper.domain.TargetTemplate
import com.direwolf.archeryhelper.domain.TenRingMode
import com.direwolf.archeryhelper.domain.TripleLayout
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ScoringEngineTest {
    private val engine: ScoringEngine = ArcheryScoringEngine()

    @Test
    fun sharedScoringFixturesMatchReferenceContract() {
        for (case in loadFixtures()) {
            val template = TargetTemplate.from(
                TargetConfig(
                    format = TargetFormat.valueOf(case["format"]!!),
                    tripleLayout = case["layout"].orEmpty().takeIf { it.isNotEmpty() }?.let(TripleLayout::valueOf),
                    minimumScoringZone = case["minimum_zone"]!!.toInt(),
                    tenRingMode = TenRingMode.valueOf(case["ten_mode"]!!),
                    faceDiameterMm = case["face_diameter_mm"]!!.toInt()
                )
            )
            val result = engine.score(
                targetTemplate = template,
                impact = ShotPoint(
                    xNorm = case["x_norm"]!!.toFloat(),
                    yNorm = case["y_norm"]!!.toFloat(),
                    faceIndex = case["face_index"]!!.toInt()
                ),
                arrowDiameterMm = case["arrow_diameter_mm"].orEmpty().takeIf { it.isNotEmpty() }?.toFloat(),
                localizationError = case["localization_error"]!!.toFloat()
            )

            assertEquals(case["name"], case["score"]!!.toInt(), result.finalCandidate)
            assertEquals(case["name"], case["is_x"]!!.toBoolean(), result.isX)
            assertEquals(case["name"], case["line_call"]!!.toBoolean(), result.lineCall)
        }
    }

    @Test
    fun coordinateConversionRoundTrips() {
        val original = ShotPoint(0.3f, -0.4f, faceIndex = 2)
        val restored = ShotPoint.fromPolar(original.radiusNorm(), original.angleDeg(), original.faceIndex)

        assertEquals(original.xNorm, restored.xNorm, 0.0001f)
        assertEquals(original.yNorm, restored.yNorm, 0.0001f)
        assertEquals(original.faceIndex, restored.faceIndex)
    }

    @Test
    fun analyzeCalculatesSeriesStatistics() {
        val target = TargetTemplate.from(
            TargetConfig(
                format = TargetFormat.SINGLE,
                tripleLayout = null,
                minimumScoringZone = 1,
                tenRingMode = TenRingMode.RECURVE,
                faceDiameterMm = 400
            )
        )
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
        val target = TargetTemplate.from(
            TargetConfig(
                format = TargetFormat.SINGLE,
                tripleLayout = null,
                minimumScoringZone = 1,
                tenRingMode = TenRingMode.RECURVE,
                faceDiameterMm = 400
            )
        )
        val stats = engine.analyze(target, emptyList())

        assertEquals(0, stats.totalScore)
        assertEquals(0.0, stats.averageScore, 0.0)
        assertEquals(emptyMap<Int, Int>(), stats.ringCounts)
    }

    private fun loadFixtures(): List<Map<String, String>> {
        val stream = checkNotNull(javaClass.classLoader?.getResourceAsStream("scoring_v2.csv"))
        val lines = stream.bufferedReader().use { it.readLines() }.filter { it.isNotBlank() }
        val headers = lines.first().split(",")
        return lines.drop(1).map { line ->
            val values = line.split(",")
            headers.indices.associate { headers[it] to values[it] }
        }
    }
}
