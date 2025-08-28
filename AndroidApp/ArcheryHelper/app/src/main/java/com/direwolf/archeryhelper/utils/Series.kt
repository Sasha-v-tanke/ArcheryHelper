package com.direwolf.archeryhelper.utils

data class Series(
    val number: Int,
    val shots: MutableList<Shot> = mutableListOf()
)