package com.direwolf.archeryhelper.utils

data class Distance(
    val date: String,
    val number: Int,
    val series: MutableList<Series> = mutableListOf(),
    val distance: Int = 50
)