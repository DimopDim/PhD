// java/com/example/museumemotionapp/utils/StorageUtils.kt
package com.example.museumemotionapp.utils

import android.content.Context
import android.os.Environment
import java.io.File

// Φάκελος χρήστη στο PUBLIC Downloads:
// /storage/emulated/0/Download/MuseumEmotion/<username>
fun getUserFolder(context: Context, username: String): File {
    val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
    return File(downloadsDir, "MuseumEmotion/$username")
}

// Root folder MuseumEmotion στο Downloads
fun getMuseumRootFolder(context: Context): File {
    val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
    return File(downloadsDir, "MuseumEmotion")
}
