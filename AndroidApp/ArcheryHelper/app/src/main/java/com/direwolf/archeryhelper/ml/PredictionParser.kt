package com.direwolf.archeryhelper.ml

data class PolarDetectedShot(
    val confidence: Float,
    val radiusNorm: Float,
    val angleDeg: Float
)

object PredictionParser {
    fun parsePolarTriplets(
        values: FloatArray,
        maxShots: Int,
        confidenceThreshold: Float
    ): List<PolarDetectedShot> {
        require(values.size >= maxShots * 3) {
            "Expected at least ${maxShots * 3} output values, got ${values.size}"
        }

        val shots = mutableListOf<PolarDetectedShot>()
        for (shotIndex in 0 until maxShots) {
            val offset = shotIndex * 3
            val confidence = values[offset]
            val radius = values[offset + 1]
            val angle = values[offset + 2]
            if (confidence >= confidenceThreshold && radius in 0f..1f && angle.isFinite()) {
                shots.add(PolarDetectedShot(confidence, radius, angle))
            }
        }
        return shots
    }
}
