package com.direwolf.archeryhelper.data.room

import com.direwolf.archeryhelper.data.ArcheryRepository
import com.direwolf.archeryhelper.domain.Distance
import com.direwolf.archeryhelper.domain.InputMode
import com.direwolf.archeryhelper.domain.Series
import com.direwolf.archeryhelper.domain.Shot
import com.direwolf.archeryhelper.domain.ShotPoint
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class RoomArcheryRepository(
    private val dao: ArcheryDao
) : ArcheryRepository {
    override suspend fun createDistance(inputMode: InputMode, distanceMeters: Int): Long {
        val now = Date()
        val formatter = SimpleDateFormat("dd.MM.yy - HH:mm", Locale.getDefault())
        val id = (getAllDistances().maxOfOrNull { it.number } ?: 0) + 1
        return dao.insertDistance(
            DistanceEntity(
                id = id.toLong(),
                date = formatter.format(now),
                createdAtMillis = now.time,
                distanceMeters = distanceMeters,
                inputMode = inputMode.name
            )
        )
    }

    suspend fun importDistance(
        id: Long,
        date: String,
        createdAtMillis: Long,
        distanceMeters: Int,
        inputMode: InputMode
    ) {
        dao.insertDistance(
            DistanceEntity(
                id = id,
                date = date,
                createdAtMillis = createdAtMillis,
                distanceMeters = distanceMeters,
                inputMode = inputMode.name
            )
        )
    }

    override suspend fun updateDistanceMeters(distanceId: Long, distanceMeters: Int) {
        dao.updateDistanceMeters(distanceId, distanceMeters)
    }

    override suspend fun getLastDistance(): Distance? {
        return dao.getLastDistance()?.toDomain()
    }

    override suspend fun getDistance(distanceId: Long): Distance? {
        return dao.getDistance(distanceId)?.toDomain()
    }

    override suspend fun getAllDistances(): List<Distance> {
        return dao.getAllDistances().map { it.toDomain() }
    }

    override suspend fun addSeries(distanceId: Long, seriesNumber: Int, shots: List<Shot>): Long {
        val seriesId = dao.insertSeries(SeriesEntity(distanceId = distanceId, number = seriesNumber))
        dao.insertShots(
            shots.map {
                ShotEntity(
                    seriesId = seriesId,
                    number = it.number,
                    result = it.result,
                    radiusNorm = it.radiusNorm,
                    angleDeg = it.angleDeg
                )
            }
        )
        return seriesId
    }

    override suspend fun clearAll() {
        dao.clearDistances()
    }

    private fun DistanceWithSeries.toDomain(): Distance {
        return Distance(
            date = distance.date,
            number = distance.id.toInt(),
            distance = distance.distanceMeters,
            series = series.sortedBy { it.series.number }.map { it.toDomain() }.toMutableList(),
            inputMode = InputMode.valueOf(distance.inputMode),
            createdAtMillis = distance.createdAtMillis
        )
    }

    private fun SeriesWithShots.toDomain(): Series {
        return Series(
            number = series.number,
            shots = shots.sortedBy { it.number }.map {
                val point = if (it.radiusNorm != null && it.angleDeg != null) {
                    ShotPoint.fromPolar(it.radiusNorm, it.angleDeg)
                } else {
                    null
                }
                Shot(
                    number = it.number,
                    result = it.result,
                    xNorm = point?.xNorm,
                    yNorm = point?.yNorm
                )
            }.toMutableList()
        )
    }
}
