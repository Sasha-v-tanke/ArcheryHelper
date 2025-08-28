package com.direwolf.archeryhelper.managers

import android.app.Application

class Application : Application() {
    val imageHolder = ImageHolder

    override fun onCreate() {
        super.onCreate()
        DataManager.init(this)
        imageHolder.getImage()
    }
}