package com.example.museumemotionapp.utils

import android.content.Context
import android.os.Environment
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.*

fun logOrUpdateUserEmotion(
    context: Context,
    username: String,
    artworkId: String,
    emotionId: String?,
    timestampEntry: Long,
    timestampExit: Long?
) {
    val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
    val userFolder = File(downloadsDir, "MuseumEmotion/$username")
    val logFile = File(userFolder, "clickOnArtwork.txt")

    // Ensure directory exists
    if (!userFolder.exists()) {
        userFolder.mkdirs()
    }

    val sdf = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
    val entryTime = sdf.format(Date(timestampEntry))
    val exitTime = timestampExit?.let { sdf.format(Date(it)) } ?: "N/A"

    if (!logFile.exists()) {
        logFile.createNewFile()
    }

    val lines = logFile.readLines().toMutableList()
    var found = false

    // Check if entry already exists and update it
    for (i in lines.indices) {
        val parts = lines[i].split(" | ")
        if (parts.size == 5 && parts[0] == username && parts[1] == artworkId) {
            // If emotion is not logged yet, update the row
            if (parts[3] == "N/A" && emotionId != null) {
                lines[i] = "${parts[0]} | ${parts[1]} | ${parts[2]} | $emotionId | $exitTime"
                found = true
                break
            }
        }
    }

    if (!found) {
        // If no entry was found, append a new one
        val logEntry = "$username | $artworkId | $entryTime | ${emotionId ?: "N/A"} | $exitTime\n"
        lines.add(logEntry)
    }

    // Write updated content back to the file
    logFile.writeText(lines.joinToString("\n"))
}

fun logAudioEmotion(
    context: Context,
    username: String,
    artworkId: String,
    emotionId: String?,
    timestampEntry: Long,
    timestampExit: Long?
) {
    val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
    val userFolder = File(downloadsDir, "MuseumEmotion/$username")
    val logFile = File(userFolder, "audioEmotionLog.txt")

    // ✅ Ensure the directory exists
    if (!userFolder.exists()) {
        userFolder.mkdirs()
    }

    val sdf = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())

    // ✅ Format timestamps correctly
    val entryTime = sdf.format(Date(timestampEntry))
    val exitTime = timestampExit?.let { sdf.format(Date(it)) } ?: "N/A"

    if (!logFile.exists()) {
        logFile.createNewFile()
    }

    val logEntry = "$username | $artworkId | $entryTime | ${emotionId ?: "N/A"} | $exitTime\n"

    try {
        FileWriter(logFile, true).use { writer ->
            writer.append(logEntry)
        }
        println("✅ LOG WRITTEN: $logEntry") // ✅ Debugging Log
    } catch (e: Exception) {
        e.printStackTrace()
        println("⚠ ERROR: Failed to write log to ${logFile.absolutePath}")
    }
}
