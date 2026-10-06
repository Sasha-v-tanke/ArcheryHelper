package com.direwolf.archeryhelper.managers

import android.content.Context
import android.content.SharedPreferences
import androidx.room.Room
import com.direwolf.archeryhelper.data.room.ArcheryDatabase
import com.direwolf.archeryhelper.data.room.RoomArcheryRepository
import com.direwolf.archeryhelper.data.room.SharedPrefsMigration
import com.direwolf.archeryhelper.utils.Distance
import com.direwolf.archeryhelper.utils.Series
import com.direwolf.archeryhelper.utils.Shot
import com.direwolf.archeryhelper.utils.debugLog
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking

object DataManager {

    private lateinit var prefs: SharedPreferences
    private lateinit var repository: RoomArcheryRepository

    fun init(context: Context) {
        prefs = context.applicationContext.getSharedPreferences("archery_data", Context.MODE_PRIVATE)
        val database = Room.databaseBuilder(
            context.applicationContext,
            ArcheryDatabase::class.java,
            "archery_data.db"
        ).build()
        repository = RoomArcheryRepository(database.archeryDao())
        runBlocking(Dispatchers.IO) {
            SharedPrefsMigration.migrate(prefs, repository)
        }
    }

    fun startNewDistance(manualInput: Boolean) {
        runBlocking(Dispatchers.IO) {
            repository.createDistance(
                if (manualInput) com.direwolf.archeryhelper.domain.InputMode.MANUAL
                else com.direwolf.archeryhelper.domain.InputMode.SCAN
            )
        }
    }

    fun getDistance(distanceIndex: Int): Int {
        return loadDistance(distanceIndex).distance
    }

    fun updateDistance(distanceIndex: Int, newDistance: Int) {
        runBlocking(Dispatchers.IO) {
            repository.updateDistanceMeters(distanceIndex.toLong(), newDistance)
        }
    }

    fun isManualInput(distanceIndex: Int): Boolean {
        return loadDistance(distanceIndex).inputMode == com.direwolf.archeryhelper.domain.InputMode.MANUAL
    }

    fun loadLastDistance(): Distance {
        return runBlocking(Dispatchers.IO) {
            repository.getLastDistance() ?: Distance("", 0, 50)
        }
    }

    fun loadDistance(distanceIndex: Int): Distance {
        return runBlocking(Dispatchers.IO) {
            repository.getDistance(distanceIndex.toLong()) ?: Distance("", distanceIndex, 50)
        }
    }

    fun getLastDistanceIndex(): Int {
        return runBlocking(Dispatchers.IO) {
            repository.getAllDistances().maxOfOrNull { it.number } ?: 0
        }
    }

    fun loadSeries(distanceIndex: Int, seriesIndex: Int): Series {
        return loadDistance(distanceIndex).series.firstOrNull { it.number == seriesIndex } ?: Series(seriesIndex)
    }

    fun getLastSeriesIndex(distanceIndex: Int = getLastDistanceIndex()): Int {
        return loadDistance(distanceIndex).series.maxOfOrNull { it.number } ?: 0
    }

    fun saveSeries(series: Series, distanceIndex: Int = getLastDistanceIndex()) {
        runBlocking(Dispatchers.IO) {
            repository.addSeries(distanceIndex.toLong(), series.number, series.shots)
        }
    }

    fun clearAllData() {
        runBlocking(Dispatchers.IO) {
            repository.clearAll()
        }
        prefs.edit().clear().apply()
    }

    fun dumpPrefs() {
        val distances = runBlocking(Dispatchers.IO) {
            repository.getAllDistances()
        }
        if (distances.isEmpty()) {
            debugLog("Данные пустые")
            return
        }

        for (distance in distances) {
            debugLog(distance.toString())
        }
    }
}

// data format
// distance_${index}: String - data
// distance_${index}_series_${index}_manual_input: Boolean
// distance_${index}_series_${seriesNumber}_shot_${shotNumber}_result: Int 0..10
// distance_${index}_series_${seriesNumber}_shot_${shotNumber}_distance: Float
// distance_${index}_series_${seriesNumber}_shot_${shotNumber}_angle: Float
