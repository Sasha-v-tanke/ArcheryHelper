package com.direwolf.archeryhelper.utils

data class Distance(
    val date: String,
    val number: Int,
    val distance: Int,
    val series: MutableList<Series> = mutableListOf()
)