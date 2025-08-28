package com.direwolf.archeryhelper.utils

data class Shot(
    var number: Int,
    val result: Int,
    val distance: Float? = null,
    val angle: Float? = null
)

