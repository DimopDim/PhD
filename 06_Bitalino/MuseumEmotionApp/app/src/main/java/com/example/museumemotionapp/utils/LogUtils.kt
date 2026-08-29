package com.example.museumemotionapp.utils

import android.content.Context
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

// Μικρό helper ώστε να μη χαλάμε το " | " delimiter και τις γραμμές
private fun sanitizeForLog(text: String): String =
    text
        .replace("\n", " ")   // σβήνουμε newlines
        .replace("|", "/")    // αντικαθιστούμε κάθε κάθετο

fun logOrUpdateUserEmotion(
    context: Context,
    username: String,
    artworkId: String,
    emotionId: String?,
    intensityLevel: Int?,
    timestampEntry: Long,
    timestampExit: Long?,
    emotionLabel: String?
) {
    // Χρήση app-specific external storage για τον συγκεκριμένο χρήστη
    val userFolder = getUserFolder(context, username)

    if (!userFolder.exists()) {
        userFolder.mkdirs()
    }

    val logFile = File(userFolder, "clickOnArtwork.txt")

    val sdf = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
    val entryTime = sdf.format(Date(timestampEntry))
    val exitTime = timestampExit?.let { sdf.format(Date(it)) } ?: "N/A"
    val intensity = intensityLevel?.toString() ?: "N/A"

    val rawLabel = emotionLabel ?: "N/A"
    val label = if (rawLabel == "N/A") rawLabel else sanitizeForLog(rawLabel)

    // Αν δεν υπάρχει αρχείο, το δημιουργούμε και γράφουμε header
    if (!logFile.exists()) {
        logFile.createNewFile()
        logFile.writeText(
            "username | artworkId | timestampEntry | emotionId | timestampExit | intensityLevel | emotionLabel\n"
        )
    }

    val lines = logFile.readLines().toMutableList()
    var found = false

    // Παράλειψη header (γραμμή 0)
    for (i in 1 until lines.size) {
        val parts = lines[i].split(" | ")
        if (parts.size >= 5 && parts[0] == username && parts[1] == artworkId) {
            // Αν το emotionId είναι ακόμα "N/A" και τώρα έχουμε τιμή, ενημερώνουμε την εγγραφή
            if (parts[3] == "N/A" && emotionId != null) {
                lines[i] =
                    "${parts[0]} | ${parts[1]} | ${parts[2]} | $emotionId | $exitTime | $intensity | $label"
                found = true
                break
            }
        }
    }

    if (!found) {
        val logEntry =
            "$username | $artworkId | $entryTime | ${emotionId ?: "N/A"} | $exitTime | $intensity | $label"
        lines.add(logEntry)
    }

    logFile.writeText(lines.joinToString("\n"))
}

fun logAudioEmotion(
    context: Context,
    username: String,
    artworkId: String,
    emotionId: String?,
    intensityLevel: Int?,
    timestampEntry: Long,
    timestampExit: Long?,
    emotionLabel: String?
) {
    val userFolder = getUserFolder(context, username)

    if (!userFolder.exists()) {
        userFolder.mkdirs()
    }

    val logFile = File(userFolder, "audioEmotionLog.txt")

    val sdf = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
    val entryTime = sdf.format(Date(timestampEntry))
    val exitTime = timestampExit?.let { sdf.format(Date(it)) } ?: "N/A"
    val intensity = intensityLevel?.toString() ?: "N/A"

    val rawLabel = emotionLabel ?: "N/A"
    val label = if (rawLabel == "N/A") rawLabel else sanitizeForLog(rawLabel)

    val header =
        "username | artworkId | timestampEntry | emotionId | timestampExit | intensityLevel | emotionLabel\n"
    val logEntry =
        "$username | $artworkId | $entryTime | ${emotionId ?: "N/A"} | $exitTime | $intensity | $label"

    if (!logFile.exists()) {
        logFile.createNewFile()
        logFile.writeText(header)
    }

    try {
        FileWriter(logFile, true).use { writer ->
            writer.append("$logEntry\n")
        }
        println("LOG WRITTEN: $logEntry")
    } catch (e: Exception) {
        e.printStackTrace()
        println("⚠ ERROR: Failed to write log to ${logFile.absolutePath}")
    }
}

fun getVisitedArtworksFromLog(context: Context, username: String): Set<String> {
    val userFolder = getUserFolder(context, username)
    val logFile = File(userFolder, "clickOnArtwork.txt")

    if (!logFile.exists()) return emptySet()

    return logFile.readLines()
        .drop(1) // skip header
        .mapNotNull { line ->
            val parts = line.split(" | ")
            if (parts.size >= 5 && parts[0] == username && parts[3] != "N/A") {
                parts[1] // artworkId
            } else {
                null
            }
        }
        .toSet()
}
