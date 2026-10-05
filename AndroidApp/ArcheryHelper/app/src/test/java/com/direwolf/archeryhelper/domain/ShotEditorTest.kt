package com.direwolf.archeryhelper.domain

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ShotEditorTest {
    @Test
    fun addMoveRemoveCanBeUndone() {
        val editor = ShotEditor(listOf(ShotPoint(0f, 0f)))

        editor.add(ShotPoint(0.1f, 0.2f))
        assertEquals(2, editor.snapshot().size)

        editor.move(1, ShotPoint(0.3f, 0.4f))
        assertEquals(ShotPoint(0.3f, 0.4f), editor.snapshot()[1])

        editor.remove(0)
        assertEquals(listOf(ShotPoint(0.3f, 0.4f)), editor.snapshot())

        assertTrue(editor.undo())
        assertEquals(listOf(ShotPoint(0f, 0f), ShotPoint(0.3f, 0.4f)), editor.snapshot())
        assertTrue(editor.undo())
        assertEquals(listOf(ShotPoint(0f, 0f), ShotPoint(0.1f, 0.2f)), editor.snapshot())
        assertTrue(editor.undo())
        assertEquals(listOf(ShotPoint(0f, 0f)), editor.snapshot())
        assertFalse(editor.undo())
    }

    @Test
    fun resetRestoresInitialShots() {
        val editor = ShotEditor(listOf(ShotPoint(0.1f, 0f)))

        editor.add(ShotPoint(0.2f, 0f))
        editor.reset()

        assertEquals(listOf(ShotPoint(0.1f, 0f)), editor.snapshot())
    }
}
