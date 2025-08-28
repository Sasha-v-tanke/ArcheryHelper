package com.direwolf.archeryhelper.managers

import android.graphics.Bitmap

object ImageHolder {
    private var image: Bitmap? = null

    fun getImage(): Bitmap? {
        return image
    }

    fun setImage(newImage: Bitmap) {
        image = newImage
    }
}