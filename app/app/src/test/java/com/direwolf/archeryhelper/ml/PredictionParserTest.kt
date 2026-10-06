package com.direwolf.archeryhelper.ml

import org.junit.Assert.assertEquals
import org.junit.Test

class PredictionParserTest {
    @Test
    fun parsePolarTripletsKeepsConfidentValidShots() {
        val shots = PredictionParser.parsePolarTriplets(
            floatArrayOf(
                1f, 0.2f, 30f,
                0.4f, 0.3f, 40f,
                0.8f, 1.2f, 50f
            ),
            maxShots = 3,
            confidenceThreshold = 0.5f
        )

        assertEquals(1, shots.size)
        assertEquals(1f, shots[0].confidence, 0.0001f)
        assertEquals(0.2f, shots[0].radiusNorm, 0.0001f)
        assertEquals(30f, shots[0].angleDeg, 0.0001f)
    }

    @Test(expected = IllegalArgumentException::class)
    fun parsePolarTripletsRejectsShortOutput() {
        PredictionParser.parsePolarTriplets(floatArrayOf(1f, 0.2f), 1, 0.5f)
    }
}
