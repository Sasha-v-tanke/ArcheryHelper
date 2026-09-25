package com.direwolf.archeryhelper.managers

import android.app.Application

class Application : Application() {
    override fun onCreate() {
        super.onCreate()
        DataManager.init(this)
    }
}
