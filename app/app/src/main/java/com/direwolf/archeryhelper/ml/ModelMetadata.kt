package com.direwolf.archeryhelper.ml

import android.content.Context
import org.json.JSONObject

data class ModelMetadata(
    val contractVersion: Int,
    val width: Int,
    val height: Int,
    val output: String,
    val maxShots: Int,
    val coordinateSystem: String,
    val confidenceThreshold: Float
) {
    companion object {
        fun load(context: Context, assetName: String = "model_metadata.json"): ModelMetadata {
            val json = context.assets.open(assetName).bufferedReader().use { it.readText() }
            val obj = JSONObject(json)
            return ModelMetadata(
                contractVersion = obj.getInt("contract_version"),
                width = obj.getInt("width"),
                height = obj.getInt("height"),
                output = obj.getString("output"),
                maxShots = obj.getInt("max_shots"),
                coordinateSystem = obj.getString("coordinate_system"),
                confidenceThreshold = obj.getDouble("confidence_threshold").toFloat()
            )
        }
    }
}
