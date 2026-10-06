package com.direwolf.archeryhelper.data.room

import android.content.SharedPreferences
import com.direwolf.archeryhelper.domain.InputMode
import com.direwolf.archeryhelper.domain.Shot

object SharedPrefsMigration {
    private const val MIGRATION_DONE = "room_migration_v1_done"

    suspend fun migrate(prefs: SharedPreferences, repository: RoomArcheryRepository) {
        if (prefs.getBoolean(MIGRATION_DONE, false)) return
        if (repository.getAllDistances().isNotEmpty()) {
            prefs.edit().putBoolean(MIGRATION_DONE, true).apply()
            return
        }

        var distanceIndex = 1
        while (prefs.contains("distance_${distanceIndex}")) {
            repository.importDistance(
                id = distanceIndex.toLong(),
                date = prefs.getString("distance_${distanceIndex}", "") ?: "",
                createdAtMillis = 0L,
                distanceMeters = prefs.getInt("distance_${distanceIndex}_distance", 50),
                inputMode = if (prefs.getBoolean("distance_${distanceIndex}_manual_input", true)) {
                    InputMode.MANUAL
                } else {
                    InputMode.SCAN
                }
            )

            var seriesIndex = 1
            while (prefs.contains("distance_${distanceIndex}_series_${seriesIndex}_shot_1_result")) {
                val shots = mutableListOf<Shot>()
                var shotIndex = 1
                while (prefs.contains("distance_${distanceIndex}_series_${seriesIndex}_shot_${shotIndex}_result")) {
                    val base = "distance_${distanceIndex}_series_${seriesIndex}_shot_${shotIndex}"
                    shots.add(
                        Shot(
                            number = shotIndex,
                            result = prefs.getInt("${base}_result", 0),
                            distance = if (prefs.contains("${base}_distance")) prefs.getFloat("${base}_distance", 0f) else null,
                            angle = if (prefs.contains("${base}_angle")) prefs.getFloat("${base}_angle", 0f) else null
                        )
                    )
                    shotIndex++
                }
                repository.addSeries(distanceIndex.toLong(), seriesIndex, shots)
                seriesIndex++
            }

            distanceIndex++
        }

        prefs.edit().putBoolean(MIGRATION_DONE, true).apply()
    }
}
