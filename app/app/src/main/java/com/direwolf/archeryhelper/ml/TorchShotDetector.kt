package com.direwolf.archeryhelper.ml

import android.content.Context
import android.graphics.Bitmap
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream

class TorchShotDetector(
    private val context: Context
) {
    private val metadata: ModelMetadata by lazy { ModelMetadata.load(context) }
    private val module: Module by lazy { LiteModuleLoader.load(assetFilePath("model.ptl")) }

    fun detect(bitmap: Bitmap): List<PolarDetectedShot> {
        if (metadata.contractVersion != 1) {
            throw IllegalStateException("Unsupported model contract: ${metadata.contractVersion}")
        }
        if (metadata.output != "shot_set_polar") {
            throw IllegalStateException("Unsupported model output: ${metadata.output}")
        }

        val input = BitmapPreprocessor.toTensor(bitmap, metadata.width, metadata.height)
        val output = module.forward(IValue.from(input)).toTensor()
        return PredictionParser.parsePolarTriplets(
            output.dataAsFloatArray,
            metadata.maxShots,
            metadata.confidenceThreshold
        )
    }

    private fun assetFilePath(assetName: String): String {
        val file = File(context.filesDir, assetName)
        context.assets.open(assetName).use { input ->
            FileOutputStream(file).use { output -> input.copyTo(output) }
        }
        return file.absolutePath
    }
}

object BitmapPreprocessor {
    fun toTensor(bitmap: Bitmap, width: Int, height: Int): Tensor {
        val scaled = if (bitmap.width == width && bitmap.height == height) {
            bitmap
        } else {
            Bitmap.createScaledBitmap(bitmap, width, height, true)
        }

        val floatBuffer = FloatArray(3 * height * width)
        val pixels = IntArray(width * height)
        scaled.getPixels(pixels, 0, width, 0, 0, width, height)

        val planeSize = height * width
        for (idx in pixels.indices) {
            val clr = pixels[idx]
            floatBuffer[idx] = ((clr shr 16) and 0xFF) / 255f
            floatBuffer[planeSize + idx] = ((clr shr 8) and 0xFF) / 255f
            floatBuffer[planeSize * 2 + idx] = (clr and 0xFF) / 255f
        }

        return Tensor.fromBlob(floatBuffer, longArrayOf(1, 3, height.toLong(), width.toLong()))
    }
}
