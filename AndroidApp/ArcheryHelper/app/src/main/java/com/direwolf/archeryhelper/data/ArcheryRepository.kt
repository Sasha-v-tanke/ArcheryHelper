package com.direwolf.archeryhelper.data

import com.direwolf.archeryhelper.domain.Distance
import com.direwolf.archeryhelper.domain.InputMode
import com.direwolf.archeryhelper.domain.Shot

interface ArcheryRepository {
    suspend fun createDistance(inputMode: InputMode, distanceMeters: Int = 50): Long
    suspend fun updateDistanceMeters(distanceId: Long, distanceMeters: Int)
    suspend fun getLastDistance(): Distance?
    suspend fun getDistance(distanceId: Long): Distance?
    suspend fun getAllDistances(): List<Distance>
    suspend fun addSeries(distanceId: Long, seriesNumber: Int, shots: List<Shot>): Long
    suspend fun clearAll()
}
