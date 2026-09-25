package com.direwolf.archeryhelper.image

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.File
import java.io.FileOutputStream

object CapturedImageRepository {
    private const val PREFS_NAME = "captured_image"
    private const val KEY_PATH = "current_path"
    private const val FILE_NAME = "current_scan.png"

    fun save(context: Context, bitmap: Bitmap): String {
        val file = File(context.cacheDir, FILE_NAME)
        FileOutputStream(file).use {
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, it)
        }
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_PATH, file.absolutePath)
            .apply()
        return file.absolutePath
    }

    fun load(context: Context): Bitmap? {
        val path = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getString(KEY_PATH, null)
            ?: return null
        val file = File(path)
        if (!file.exists()) return null
        return BitmapFactory.decodeFile(file.absolutePath)
    }

    fun clear(context: Context) {
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        prefs.getString(KEY_PATH, null)?.let { File(it).delete() }
        prefs.edit().clear().apply()
    }
}
