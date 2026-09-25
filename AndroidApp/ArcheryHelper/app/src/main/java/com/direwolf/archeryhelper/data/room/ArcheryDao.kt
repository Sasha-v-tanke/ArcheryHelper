package com.direwolf.archeryhelper.data.room

import androidx.room.Dao
import androidx.room.Embedded
import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.Relation
import androidx.room.Transaction

@Entity(tableName = "distances")
data class DistanceEntity(
    @PrimaryKey val id: Long,
    val date: String,
    val createdAtMillis: Long,
    val distanceMeters: Int,
    val inputMode: String
)

@Entity(
    tableName = "series",
    foreignKeys = [
        ForeignKey(
            entity = DistanceEntity::class,
            parentColumns = ["id"],
            childColumns = ["distanceId"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [Index("distanceId")]
)
data class SeriesEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val distanceId: Long,
    val number: Int
)

@Entity(
    tableName = "shots",
    foreignKeys = [
        ForeignKey(
            entity = SeriesEntity::class,
            parentColumns = ["id"],
            childColumns = ["seriesId"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [Index("seriesId")]
)
data class ShotEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val seriesId: Long,
    val number: Int,
    val result: Int,
    val radiusNorm: Float?,
    val angleDeg: Float?
)

data class SeriesWithShots(
    @Embedded val series: SeriesEntity,
    @Relation(
        parentColumn = "id",
        entityColumn = "seriesId"
    )
    val shots: List<ShotEntity>
)

data class DistanceWithSeries(
    @Embedded val distance: DistanceEntity,
    @Relation(
        entity = SeriesEntity::class,
        parentColumn = "id",
        entityColumn = "distanceId"
    )
    val series: List<SeriesWithShots>
)

@Dao
interface ArcheryDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertDistance(entity: DistanceEntity): Long

    @Insert
    suspend fun insertSeries(entity: SeriesEntity): Long

    @Insert
    suspend fun insertShots(entities: List<ShotEntity>)

    @Query("UPDATE distances SET distanceMeters = :distanceMeters WHERE id = :distanceId")
    suspend fun updateDistanceMeters(distanceId: Long, distanceMeters: Int)

    @Transaction
    @Query("SELECT * FROM distances WHERE id = :distanceId")
    suspend fun getDistance(distanceId: Long): DistanceWithSeries?

    @Transaction
    @Query("SELECT * FROM distances ORDER BY id DESC LIMIT 1")
    suspend fun getLastDistance(): DistanceWithSeries?

    @Transaction
    @Query("SELECT * FROM distances ORDER BY id ASC")
    suspend fun getAllDistances(): List<DistanceWithSeries>

    @Query("DELETE FROM distances")
    suspend fun clearDistances()
}
