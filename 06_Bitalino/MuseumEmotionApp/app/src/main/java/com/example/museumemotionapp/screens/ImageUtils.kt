package com.example.museumemotionapp.screens

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.size
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue          // ✅ για το "by remember"
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue        // ✅
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.unit.dp
import java.io.IOException

@Composable
fun ImageFromAssets(
    context: Context,
    artworkId: String
) {
    var bitmap by remember(artworkId) { mutableStateOf<Bitmap?>(null) }

    LaunchedEffect(artworkId) {
        bitmap = try {
            // Φορτώνουμε το asset και κλείνουμε σωστά το stream
            context.assets.open("images/$artworkId.jpg").use { inputStream ->
                BitmapFactory.decodeStream(inputStream)
            }
        } catch (e: IOException) {
            // Αν δεν βρεθεί η εικόνα ή υπάρχει σφάλμα, απλά δεν δείχνουμε τίποτα
            null
        }
    }

    bitmap?.let {
        Image(
            bitmap = it.asImageBitmap(),
            contentDescription = "Artwork Image",
            modifier = Modifier.size(200.dp)
        )
    }
}
