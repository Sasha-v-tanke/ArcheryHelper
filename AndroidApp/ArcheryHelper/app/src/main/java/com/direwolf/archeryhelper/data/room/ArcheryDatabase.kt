package com.direwolf.archeryhelper.data.room

import androidx.room.Database
import androidx.room.RoomDatabase

@Database(
    entities = [
        DistanceEntity::class,
        SeriesEntity::class,
        ShotEntity::class
    ],
    version = 1,
    exportSchema = false
)
abstract class ArcheryDatabase : RoomDatabase() {
    abstract fun archeryDao(): ArcheryDao
}
