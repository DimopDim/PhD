package com.example.museumemotionapp.screens

import android.media.MediaPlayer
import android.util.Log
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController
import com.example.museumemotionapp.models.Artwork
import com.example.museumemotionapp.models.emotions
import com.example.museumemotionapp.utils.FileUtils.getAudioUrlFromWebpage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun ArtworkDetailScreen(artwork: Artwork, navController: NavController, username: String) {
    var selectedEmotion by remember { mutableStateOf<String?>(null) }
    var isEmotionSaved by remember { mutableStateOf(false) }
    var audioUrl by remember { mutableStateOf<String?>(null) }
    var isPlaying by remember { mutableStateOf(false) }
    var isPaused by remember { mutableStateOf(false) }
    var remainingTime by remember { mutableStateOf("00:00") } // Remaining time state

    val coroutineScope = rememberCoroutineScope()
    val mediaPlayer = remember { mutableStateOf<MediaPlayer?>(null) }
    var currentPosition by remember { mutableStateOf(0) } // Track current position

    val entryTime = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date())

    // Fetch Audio URL when the screen loads
    LaunchedEffect(artwork.url) {
        coroutineScope.launch(Dispatchers.IO) {
            val extractedAudioUrl = getAudioUrlFromWebpage(artwork.url)
            if (extractedAudioUrl != null) {
                audioUrl = extractedAudioUrl
                Log.d("ArtworkDetailScreen", "✅ Extracted Audio URL: $extractedAudioUrl")
            } else {
                Log.e("ArtworkDetailScreen", "⚠️ No audio found!")
            }
        }
    }

    // Monitor remaining time while playing
    LaunchedEffect(isPlaying) {
        while (isPlaying) {
            mediaPlayer.value?.let { player ->
                if (player.isPlaying) {
                    remainingTime = formatTime(player.duration - player.currentPosition)
                }
            }
            delay(500) // Update every 500ms
        }
    }

    // Clean up MediaPlayer when leaving the screen
    DisposableEffect(Unit) {
        onDispose {
            mediaPlayer.value?.release()
            mediaPlayer.value = null
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(artwork.title, style = MaterialTheme.typography.headlineMedium)
        Text(artwork.greekTitle, style = MaterialTheme.typography.bodyLarge)

        Spacer(modifier = Modifier.height(16.dp))

        // 🎧 Audio Player Controls
        if (audioUrl != null) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                Button(onClick = {
                    if (!isPlaying) {
                        mediaPlayer.value?.release()
                        mediaPlayer.value = MediaPlayer().apply {
                            setDataSource(audioUrl)
                            setOnPreparedListener {
                                if (isPaused) {
                                    seekTo(currentPosition) // Resume from last position
                                }
                                start()
                                isPlaying = true
                                isPaused = false
                            }
                            setOnCompletionListener {
                                isPlaying = false
                                isPaused = false
                                remainingTime = "00:00"
                            }
                            prepareAsync()
                        }
                    }

                    // ✅ Log when user presses Play
                    com.example.museumemotionapp.utils.FileUtils.logSoundRecorder(
                        username = username,
                        artworkId = artwork.id,
                        artworkTitle = artwork.title,
                        action = "Play"
                    )

                }) {
                    Text("Play")
                }

                Button(onClick = {
                    mediaPlayer.value?.let { player ->
                        if (player.isPlaying) {
                            currentPosition = player.currentPosition // Save position
                            player.pause()
                            isPlaying = false
                            isPaused = true
                        }
                    }

                    // ✅ Log when user presses Pause
                    com.example.museumemotionapp.utils.FileUtils.logSoundRecorder(
                        username = username,
                        artworkId = artwork.id,
                        artworkTitle = artwork.title,
                        action = "Pause"
                    )

                }) {
                    Text("Pause")
                }

                Button(onClick = {
                    mediaPlayer.value?.stop()
                    mediaPlayer.value?.release()
                    mediaPlayer.value = null
                    isPlaying = false
                    isPaused = false
                    remainingTime = "00:00"
                    currentPosition = 0

                    // ✅ Log when user presses Stop
                    com.example.museumemotionapp.utils.FileUtils.logSoundRecorder(
                        username = username,
                        artworkId = artwork.id,
                        artworkTitle = artwork.title,
                        action = "Stop"
                    )

                }) {
                    Text("Stop")
                }
            }

            // ⏳ Display remaining time
            Text("($remainingTime)", style = MaterialTheme.typography.bodyMedium)
        } else {
            Text("🔄 Loading audio...")
        }

        Spacer(modifier = Modifier.height(16.dp))

        // 🎭 Emotion Selection
        Text("Select Emotion | Επιλογή Συναισθήματος")
        LazyColumn(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
        ) {
            items(emotions) { emotion ->
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    RadioButton(
                        selected = (selectedEmotion == emotion.id),
                        onClick = {
                            selectedEmotion = emotion.id
                            isEmotionSaved = false
                        }
                    )
                    Text(text = "${emotion.id}. ${emotion.greekLabel}")
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // ✅ Save Emotion Button
        Button(
            onClick = { isEmotionSaved = true },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Save Emotion | Αποθήκευση Συναισθήματος")
        }

        if (isEmotionSaved) {
            Text("Saved | Αποθηκεύτηκε", color = MaterialTheme.colorScheme.primary)
        }

        Spacer(modifier = Modifier.height(16.dp))

        // 🚪 Exit Button - Logs the artwork click
        Button(
            onClick = {
                val exitTime = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date())

                // ✅ Log the artwork interaction with artwork ID
                com.example.museumemotionapp.utils.FileUtils.logArtworkClick(
                    username = username,
                    artworkId = artwork.id,
                    artworkTitle = artwork.title,
                    selectedEmotion = selectedEmotion,
                    entryTime = entryTime,
                    exitTime = exitTime
                )

                mediaPlayer.value?.release()
                mediaPlayer.value = null
                isPlaying = false
                isPaused = false
                navController.popBackStack()
            },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Exit | Έξοδος")
        }
    }
}

/**
 * Converts milliseconds to a human-readable format (mm:ss).
 */
fun formatTime(ms: Int): String {
    val seconds = ms / 1000
    val minutes = seconds / 60
    val remainingSeconds = seconds % 60
    return String.format("%02d:%02d", minutes, remainingSeconds)
}
