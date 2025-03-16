package com.example.museumemotionapp.screens

import android.media.MediaPlayer
import android.util.Log
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController
import com.example.museumemotionapp.models.Emotion
import com.example.museumemotionapp.models.emotions
import com.example.museumemotionapp.utils.logAudioEmotion
import kotlinx.coroutines.delay
import java.io.IOException
import java.util.concurrent.TimeUnit

@Composable
fun AudioPlaybackScreen(navController: NavController, artworkId: String, username: String) {
    val context = LocalContext.current
    val timestampEntry = remember { System.currentTimeMillis() }
    var selectedEmotion by remember { mutableStateOf<Emotion?>(null) }
    var mediaPlayer by remember { mutableStateOf<MediaPlayer?>(null) }
    var isPlaying by remember { mutableStateOf(false) }
    var currentPosition by remember { mutableStateOf(0) }
    var duration by remember { mutableStateOf(1) }
    val coroutineScope = rememberCoroutineScope()
    var showNoAudioDialog by remember { mutableStateOf(false) }

    // Disable Back Button
    BackHandler { /* Prevents back navigation manually */ }

    // Check if the audio file exists
    LaunchedEffect(Unit) {
        try {
            context.assets.openFd("audio/$artworkId.mp3").close()
        } catch (e: IOException) {
            Log.e("AudioPlaybackScreen", "Audio file not found: audio/$artworkId.mp3")
            showNoAudioDialog = true
        }
    }

    // If no audio exists, show an alert and return to previous screen
    if (showNoAudioDialog) {
        AlertDialog(
            onDismissRequest = {
                navController.navigate("artworkSelection/$username") {
                    popUpTo("artworkSelection/$username") { inclusive = true }
                }
            },
            title = { Text("Audio Not Available | Ήχος Μη Διαθέσιμος") },
            text = { Text("There is no audio file for this artwork. | Το αρχείο ήχου δεν είναι διαθέσιμο γι' αυτό το έργο.") },
            confirmButton = {
                Button(onClick = {
                    navController.navigate("artworkSelection/$username") {
                        popUpTo("artworkSelection/$username") { inclusive = true }
                    }
                }) {
                    Text("OK")
                }
            }
        )
        return // Prevents further UI rendering
    }

    // Create MediaPlayer only if the file exists
    DisposableEffect(Unit) {
        try {
            val assetFileDescriptor = context.assets.openFd("audio/$artworkId.mp3")
            val player = MediaPlayer().apply {
                setDataSource(
                    assetFileDescriptor.fileDescriptor,
                    assetFileDescriptor.startOffset,
                    assetFileDescriptor.length
                )
                prepare()
                start()
                isPlaying = true
                duration = this.duration
            }
            mediaPlayer = player

            onDispose {
                player.release()
            }
        } catch (e: IOException) {
            Log.e("AudioPlaybackScreen", "Failed to load audio", e)
        }

        onDispose {
            mediaPlayer?.release()
            mediaPlayer = null
        }
    }

    // **🔄 Restart progress update when playback state changes**
    LaunchedEffect(isPlaying) {
        while (isPlaying) {
            delay(500)
            mediaPlayer?.let {
                currentPosition = it.currentPosition
                duration = it.duration
            }
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(text = "Listening to: $artworkId")
        Text(text = "User: $username")

        Spacer(modifier = Modifier.height(16.dp))

        // Play, Pause, Stop Buttons
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.Center
        ) {
            Button(onClick = {
                mediaPlayer?.start()
                isPlaying = true
            }, enabled = mediaPlayer != null && !isPlaying) {
                Text("▶ Play")
            }

            Spacer(modifier = Modifier.width(8.dp))

            Button(onClick = {
                mediaPlayer?.pause()
                isPlaying = false
            }, enabled = mediaPlayer != null && isPlaying) {
                Text("⏸ Pause")
            }

            Spacer(modifier = Modifier.width(8.dp))

            Button(onClick = {
                mediaPlayer?.stop()
                mediaPlayer?.prepare()
                isPlaying = false
                currentPosition = 0 // **Reset progress bar**
            }, enabled = mediaPlayer != null) {
                Text("⏹ Stop")
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        // Seek -10s and +10s
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.Center
        ) {
            Button(onClick = {
                mediaPlayer?.seekTo((mediaPlayer?.currentPosition ?: 0) - 10000)
            }, enabled = mediaPlayer != null) {
                Text("⏪ -10s")
            }

            Spacer(modifier = Modifier.width(16.dp))

            Button(onClick = {
                mediaPlayer?.seekTo((mediaPlayer?.currentPosition ?: 0) + 10000)
            }, enabled = mediaPlayer != null) {
                Text("⏩ +10s")
            }
        }

        Spacer(modifier = Modifier.height(16.dp))
        Text("Time: ${formatTime(currentPosition)} / ${formatTime(duration)}")

        // Progress Bar
        Slider(
            value = currentPosition.toFloat(),
            onValueChange = {
                mediaPlayer?.seekTo(it.toInt())
                currentPosition = it.toInt()
            },
            valueRange = 0f..duration.toFloat(),
            modifier = Modifier.fillMaxWidth(),
            enabled = mediaPlayer != null
        )

        Spacer(modifier = Modifier.height(16.dp))

        Text(text = "Select how you feel while listening:")
        Text(text = "Επιλέξτε πως νιώθετε ενώ ακούτε την παρουσίαση:")

        // Emotion Selection
        Column(modifier = Modifier.weight(1f)) {
            LazyColumn {
                items(emotions) { emotion ->
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(8.dp)
                            .clickable { selectedEmotion = emotion }
                    ) {
                        RadioButton(
                            selected = selectedEmotion == emotion,
                            onClick = { selectedEmotion = emotion }
                        )
                        Column(modifier = Modifier.padding(start = 8.dp)) {
                            Text(text = "${emotion.id}. ${emotion.englishLabel}")
                            Text(text = emotion.greekLabel, style = MaterialTheme.typography.bodySmall)
                        }
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Save & Exit Button
        Button(
            onClick = {
                val timestampExit = System.currentTimeMillis()
                selectedEmotion?.let {
                    logAudioEmotion(context, username, artworkId, it.id, timestampEntry, timestampExit)
                }
                mediaPlayer?.release()
                mediaPlayer = null

                navController.navigate("artworkSelection/$username") {
                    popUpTo("artworkSelection/$username") { inclusive = true }
                }
            },
            enabled = selectedEmotion != null
        ) {
            Text("Save & Exit | Αποθήκευση & Έξοδος")
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Footer (Copyright Text)
        Text(
            text = "© 2025 MMAI Team | University of the Aegean",
            color = Color.Gray,
            textAlign = TextAlign.Center,
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 8.dp)
        )
    }
}

// Helper function to format time
fun formatTime(milliseconds: Int): String {
    val minutes = TimeUnit.MILLISECONDS.toMinutes(milliseconds.toLong())
    val seconds = TimeUnit.MILLISECONDS.toSeconds(milliseconds.toLong()) % 60
    return String.format("%02d:%02d", minutes, seconds)
}
