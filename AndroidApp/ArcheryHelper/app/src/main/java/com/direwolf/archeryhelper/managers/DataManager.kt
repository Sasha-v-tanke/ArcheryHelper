package com.direwolf.archeryhelper.managers

import android.content.Context
import android.content.SharedPreferences
import com.direwolf.archeryhelper.utils.Distance
import com.direwolf.archeryhelper.utils.Series
import com.direwolf.archeryhelper.utils.Shot
import com.direwolf.archeryhelper.utils.debugLog
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object DataManager {

    private lateinit var prefs: SharedPreferences

    fun init(context: Context) {
        prefs = context.applicationContext.getSharedPreferences("archery_data", Context.MODE_PRIVATE)
    }

    fun startNewDistance(manualInput: Boolean) {
        val editor = prefs.edit()
        val index = getLastDistanceIndex() + 1
        val now = Date()
        val formatter = SimpleDateFormat("dd.MM.yy - HH:mm", Locale.getDefault())
        editor.putString("distance_${index}", formatter.format(now))
        editor.putInt("distance_${index}_distance", 50)
        editor.putBoolean("distance_${index}_manual_input", manualInput)
        editor.apply()
    }

    fun getDistance(distanceIndex: Int): Int {
        return prefs.getInt("distance_${distanceIndex}_distance", 50)
    }

    fun updateDistance(distanceIndex: Int, newDistance: Int) {
        val editor = prefs.edit()
        editor.putInt("distance_${distanceIndex}_distance", newDistance)
        editor.apply()
    }

    fun isManualInput(distanceIndex: Int): Boolean {
        return prefs.getBoolean("distance_${distanceIndex}_manual_input", true)
    }

    fun loadLastDistance(): Distance {
        val distanceIndex = getLastDistanceIndex()
        return loadDistance(distanceIndex)
    }

    fun loadDistance(distanceIndex: Int): Distance {
        val date = prefs.getString("distance_${distanceIndex}", "") ?: ""
        val dist = prefs.getInt("distance_${distanceIndex}_distance", 50)
        val distance = Distance(date, distanceIndex, dist)
        var seriesIndex = 1
        while (prefs.contains("distance_${distanceIndex}_series_${seriesIndex}_shot_1_result")) {
            distance.series.add(loadSeries(distanceIndex, seriesIndex))
            seriesIndex++
        }
        return distance
    }

    fun getLastDistanceIndex(): Int {
        var idx = 1
        while (prefs.contains("distance_${idx}")) idx++
        return idx - 1
    }

    fun loadSeries(distanceIndex: Int, seriesIndex: Int): Series {
        val series = Series(seriesIndex)
        var shotIndex = 1
        while (prefs.contains("distance_${distanceIndex}_series_${seriesIndex}_shot_${shotIndex}_result")) {
            series.shots.add(loadShot(distanceIndex, seriesIndex, shotIndex))
            shotIndex++
        }
        return series
    }

    fun getLastSeriesIndex(distanceIndex: Int = getLastDistanceIndex()): Int {
        var idx = 1
        while (prefs.contains("distance_${distanceIndex}_series_${idx}_shot_1_result")) idx++
        return idx - 1
    }

    fun saveSeries(series: Series, distanceIndex: Int = getLastDistanceIndex()) {
        for (shot in series.shots) {
            saveShot(distanceIndex, series.number, shot)
        }
    }

    private fun saveShot(distanceIndex: Int = getLastDistanceIndex(), seriesNumber: Int, shot: Shot) {
        val editor = prefs.edit()
        val base = "distance_${distanceIndex}_series_${seriesNumber}_shot_${shot.number}"
        editor.putInt("${base}_result", shot.result)
        shot.distance?.let { editor.putFloat("${base}_distance", it) }
        shot.angle?.let { editor.putFloat("${base}_angle", it) }
        editor.apply()
    }

    private fun loadShot(distanceIndex: Int = getLastDistanceIndex(), seriesIndex: Int, shotIndex: Int): Shot {
        val base = "distance_${distanceIndex}_series_${seriesIndex}_shot_${shotIndex}"
        val result = prefs.getInt("${base}_result", 0)
        val distance = if (prefs.contains("${base}_distance")) prefs.getFloat("${base}_distance", 0f) else null
        val angle = if (prefs.contains("${base}_angle")) prefs.getFloat("${base}_angle", 0f) else null
        return Shot(shotIndex, result, distance, angle)
    }

    fun clearAllData() {
        prefs.edit().clear().apply()
    }

    fun dumpPrefs() {
        val allPrefs: Map<String, *> = prefs.all
        if (allPrefs.isEmpty()) {
            debugLog("Prefs пустые")
            return
        }

        for ((key, value) in allPrefs) {
            debugLog("$key\t=\t$value")
        }
    }
}

// data format
// distance_${index}: String - data
// distance_${index}_series_${index}_manual_input: Boolean
// distance_${index}_series_${seriesNumber}_shot_${shotNumber}_result: Int 0..10
// distance_${index}_series_${seriesNumber}_shot_${shotNumber}_distance: Float
// distance_${index}_series_${seriesNumber}_shot_${shotNumber}_angle: Float