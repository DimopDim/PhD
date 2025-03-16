package com.example.museumemotionapp.screens

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.os.Build
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.size
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.unit.dp
import java.io.IOException

@Composable
fun ImageFromAssets(context: Context, artworkId: String) {
    var bitmap by remember { mutableStateOf<Bitmap?>(null) }

    LaunchedEffect(artworkId) {
        try {
            val inputStream = context.assets.open("images/$artworkId.jpg")
            bitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                val source = ImageDecoder.createSource(context.assets, "images/$artworkId.jpg")
                ImageDecoder.decodeBitmap(source)
            } else {
                BitmapFactory.decodeStream(inputStream)
            }
        } catch (e: IOException) {
            bitmap = null // Handle missing image
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
