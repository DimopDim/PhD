package com.example.museumemotionapp.utils

import android.os.Environment
import android.util.Log
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileWriter
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.regex.Pattern

object FileUtils {
    private val client = OkHttpClient()

    /**
     * Fetches an audio URL from a given artwork webpage.
     */
    fun getAudioUrlFromWebpage(pageUrl: String): String? {
        return try {
            val request = Request.Builder().url(pageUrl).build()
            val response = client.newCall(request).execute()
            val html = response.body?.string() ?: return null

            // Extract audio file URL using regex
            val pattern = Pattern.compile("https://.*?\\.mp3")
            val matcher = pattern.matcher(html)

            if (matcher.find()) {
                matcher.group(0)  // Return the first MP3 URL found
            } else {
                null
            }
        } catch (e: Exception) {
            Log.e("FileUtils", "Failed to fetch audio URL: ${e.message}")
            null
        }
    }

    /**
     * Logs when a user logs into the app.
     */
    fun logUserLogin(username: String) {
        val userFolder = getUserFolder(username)
        val logFile = File(userFolder, "userLog.txt")

        try {
            ensureFolderExists(userFolder)

            val timestamp = getCurrentTimestamp()
            val logEntry = "$username - $timestamp\n"

            FileWriter(logFile, true).use { writer ->
                writer.append(logEntry)
            }

            Log.d("FileUtils", "✅ User login saved: $logEntry")

        } catch (e: IOException) {
            Log.e("FileUtils", "❌ Error writing user login log: ${e.message}")
        }
    }

    /**
     * Logs when a user selects an artwork and their emotion.
     */
    fun logArtworkClick(
        username: String,
        artworkId: String,
        artworkTitle: String,
        selectedEmotion: String?,
        entryTime: String,
        exitTime: String
    ) {
        val userFolder = getUserFolder(username)
        val logFile = File(userFolder, "clickOnArtwork.txt")

        try {
            ensureFolderExists(userFolder)

            val logEntry =
                "Artwork ID: $artworkId | Artwork: $artworkTitle | Entry: $entryTime | Emotion: ${selectedEmotion ?: "None"} | Exit: $exitTime\n"

            FileWriter(logFile, true).use { writer ->
                writer.append(logEntry)
            }

            Log.d("FileUtils", "✅ Artwork log saved: $logEntry")

        } catch (e: IOException) {
            Log.e("FileUtils", "❌ Error writing artwork log: ${e.message}")
        }
    }

    /**
     * Logs Play and Stop times in the same line for the audio player.
     */
    fun logSoundRecorder(username: String, artworkId: String, artworkTitle: String, action: String) {
        val userFolder = getUserFolder(username)
        val logFile = File(userFolder, "soundRecorder.txt")

        try {
            ensureFolderExists(userFolder)

            val timestamp = getCurrentTimestamp()

            if (action == "Play") {
                // Save a new line for Play event with start time
                val logEntry = "Artwork ID: $artworkId | Artwork: $artworkTitle | Start: $timestamp | Stop: -\n"
                FileWriter(logFile, true).use { writer ->
                    writer.append(logEntry)
                }
                Log.d("FileUtils", "✅ SoundRecorder log saved (Play): $logEntry")
            } else if (action == "Stop") {
                // Read the file, find the last "Start" line, and update it with Stop time
                val lines = logFile.readLines().toMutableList()
                for (i in lines.indices.reversed()) {
                    if (lines[i].contains("Start:") && lines[i].contains("Stop: -")) {
                        lines[i] = lines[i].replace("Stop: -", "Stop: $timestamp")
                        break
                    }
                }

                // Write updated content back to the file
                FileWriter(logFile, false).use { writer ->
                    lines.forEach { writer.append(it).append("\n") }
                }
                Log.d("FileUtils", "✅ SoundRecorder log updated (Stop): $timestamp")
            }

        } catch (e: IOException) {
            Log.e("FileUtils", "❌ Error writing soundRecorder log: ${e.message}")
        }
    }

    /**
     * Returns the user's folder path inside `Downloads/MuseumEmotion/`.
     */
    private fun getUserFolder(username: String): File {
        val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
        return File(downloadsDir, "MuseumEmotion/$username")
    }

    /**
     * Ensures that the user's folder exists before writing logs.
     */
    private fun ensureFolderExists(folder: File) {
        if (!folder.exists()) {
            val folderCreated = folder.mkdirs()
            if (!folderCreated) {
                Log.e("FileUtils", "⚠️ Failed to create user folder: $folder")
            }
        }
    }

    /**
     * Returns the current timestamp in `yyyy-MM-dd HH:mm:ss` format.
     */
    private fun getCurrentTimestamp(): String {
        return SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date())
    }
}
