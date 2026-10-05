package com.direwolf.archeryhelper.domain

class ShotEditor(initialShots: List<ShotPoint> = emptyList()) {
    private val initial = initialShots.toList()
    private val history = mutableListOf<List<ShotPoint>>()
    private val shots = initialShots.toMutableList()

    fun add(point: ShotPoint) {
        saveHistory()
        shots.add(point)
    }

    fun move(index: Int, point: ShotPoint) {
        if (index !in shots.indices) return
        saveHistory()
        shots[index] = point
    }

    fun remove(index: Int) {
        if (index !in shots.indices) return
        saveHistory()
        shots.removeAt(index)
    }

    fun undo(): Boolean {
        val previous = history.removeLastOrNull() ?: return false
        shots.clear()
        shots.addAll(previous)
        return true
    }

    fun reset() {
        saveHistory()
        shots.clear()
        shots.addAll(initial)
    }

    fun snapshot(): List<ShotPoint> = shots.toList()

    private fun saveHistory() {
        history.add(shots.toList())
    }
}
