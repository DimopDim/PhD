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
import androidx.compose.ui.unit.sp
import androidx.navigation.NavController
import com.example.museumemotionapp.LocalFontScale
import com.example.museumemotionapp.models.Emotion
import com.example.museumemotionapp.models.emotions
import com.example.museumemotionapp.utils.logAudioEmotion
import kotlinx.coroutines.delay
import java.io.IOException
import java.util.concurrent.TimeUnit
import kotlin.math.roundToInt

@Composable
fun AudioPlaybackScreen(navController: NavController, artworkId: String, username: String) {
    val context = LocalContext.current
    val scale = LocalFontScale.current.scale
    val timestampEntry = remember { System.currentTimeMillis() }
    var selectedEmotion by remember { mutableStateOf<Emotion?>(null) }
    var intensityLevel by remember { mutableStateOf(4f) }
    var sliderTouched by remember { mutableStateOf(false) }
    var mediaPlayer by remember { mutableStateOf<MediaPlayer?>(null) }
    var isPlaying by remember { mutableStateOf(false) }
    var currentPosition by remember { mutableStateOf(0) }
    var duration by remember { mutableStateOf(1) }
    var showNoAudioDialog by remember { mutableStateOf(false) }

    // For custom "Other" emotion
    var showCustomEmotionDialog by remember { mutableStateOf(false) }
    var customEmotionText by remember { mutableStateOf("") }

    BackHandler { }

    LaunchedEffect(Unit) {
        try {
            context.assets.openFd("audio/$artworkId.mp3").close()
        } catch (e: IOException) {
            Log.e("AudioPlaybackScreen", "Audio file not found: audio/$artworkId.mp3")
            showNoAudioDialog = true
        }
    }

    if (showNoAudioDialog) {
        AlertDialog(
            onDismissRequest = {
                navController.navigate("artworkSelection/$username") {
                    popUpTo("artworkSelection/$username") { inclusive = true }
                }
            },
            title = { Text("Audio Not Available | Ήχος Μη Διαθέσιμος", fontSize = 18.sp * scale) },
            text = {
                Text(
                    "There is no audio file for this artwork. | Το αρχείο ήχου δεν είναι διαθέσιμο γι' αυτό το έργο.",
                    fontSize = 14.sp * scale
                )
            },
            confirmButton = {
                Button(onClick = {
                    navController.navigate("artworkSelection/$username") {
                        popUpTo("artworkSelection/$username") { inclusive = true }
                    }
                }) {
                    Text("OK", fontSize = 16.sp * scale)
                }
            }
        )
        return
    }

    DisposableEffect(Unit) {
        try {
            val afd = context.assets.openFd("audio/$artworkId.mp3")
            val player = MediaPlayer().apply {
                setDataSource(afd.fileDescriptor, afd.startOffset, afd.length)
                prepare()
                start()
                isPlaying = true
                duration = this.duration
            }
            mediaPlayer = player

            onDispose { player.release() }
        } catch (e: IOException) {
            Log.e("AudioPlaybackScreen", "Failed to load audio", e)
        }

        onDispose {
            mediaPlayer?.release()
            mediaPlayer = null
        }
    }

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
        modifier = Modifier.fillMaxSize().padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text("Listening to: $artworkId", fontSize = 16.sp * scale)
        Text("User: $username", fontSize = 16.sp * scale)

        Spacer(modifier = Modifier.height(16.dp))

        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Center) {
            Button(onClick = {
                mediaPlayer?.start(); isPlaying = true
            }, enabled = mediaPlayer != null && !isPlaying) {
                Text("▶ Play", fontSize = 16.sp * scale)
            }

            Spacer(modifier = Modifier.width(8.dp))

            Button(onClick = {
                mediaPlayer?.pause(); isPlaying = false
            }, enabled = mediaPlayer != null && isPlaying) {
                Text("⏸ Pause", fontSize = 16.sp * scale)
            }

            Spacer(modifier = Modifier.width(8.dp))

            Button(onClick = {
                mediaPlayer?.stop()
                mediaPlayer?.prepare()
                isPlaying = false
                currentPosition = 0
            }, enabled = mediaPlayer != null) {
                Text("⏹ Stop", fontSize = 16.sp * scale)
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Center) {
            Button(onClick = {
                mediaPlayer?.seekTo((mediaPlayer?.currentPosition ?: 0) - 10000)
            }, enabled = mediaPlayer != null) {
                Text("⏪ -10s", fontSize = 16.sp * scale)
            }

            Spacer(modifier = Modifier.width(16.dp))

            Button(onClick = {
                mediaPlayer?.seekTo((mediaPlayer?.currentPosition ?: 0) + 10000)
            }, enabled = mediaPlayer != null) {
                Text("⏩ +10s", fontSize = 16.sp * scale)
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        Text("Time: ${formatTime(currentPosition)} / ${formatTime(duration)}", fontSize = 14.sp * scale)

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

        Text("Select how you feel while listening:", fontSize = 16.sp * scale)
        Text("Επιλέξτε πως νιώθετε ενώ ακούτε την παρουσίαση:", fontSize = 16.sp * scale)

        Column(modifier = Modifier.weight(1f)) {
            LazyColumn {
                items(emotions) { emotion ->
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(8.dp)
                            .clickable {
                                selectedEmotion = emotion
                                if (emotion.id.toIntOrNull() == 23) {
                                    showCustomEmotionDialog = true
                                    customEmotionText = ""
                                }
                            }
                    ) {
                        RadioButton(
                            selected = selectedEmotion == emotion,
                            onClick = {
                                selectedEmotion = emotion
                                if (emotion.id.toIntOrNull() == 23) {
                                    showCustomEmotionDialog = true
                                    customEmotionText = ""
                                }
                            }
                        )
                        Column(modifier = Modifier.padding(start = 8.dp)) {
                            Text("${emotion.id}. ${emotion.englishLabel}", fontSize = 16.sp * scale)
                            Text(emotion.greekLabel, style = MaterialTheme.typography.bodySmall.copy(fontSize = 14.sp * scale))
                        }
                    }
                }
            }
        }

        if (showCustomEmotionDialog) {
            AlertDialog(
                onDismissRequest = { showCustomEmotionDialog = false },
                confirmButton = {
                    TextButton(onClick = { showCustomEmotionDialog = false }) {
                        Text("OK")
                    }
                },
                title = { Text("Περιγράψτε το συναίσθημα") },
                text = {
                    OutlinedTextField(
                        value = customEmotionText,
                        onValueChange = { customEmotionText = it },
                        label = { Text("Πληκτρολογίστε αυτό που νιώθετε") },
                        singleLine = false,
                        modifier = Modifier.fillMaxWidth()
                    )
                }
            )
        }

        Spacer(modifier = Modifier.height(16.dp))

        val intensityDescriptions = mapOf(
            1 to "Not at all | Καθόλου",
            2 to "Very little | Πολύ λίγο",
            3 to "A little | Λίγο",
            4 to "Neutral | Ουδέτερα",
            5 to "Quite a lot | Αρκετά",
            6 to "Very much | Πολύ",
            7 to "Extremely | Εξαιρετικά"
        )
        val roundedIntensity = intensityLevel.roundToInt().coerceIn(1, 7)
        val description = intensityDescriptions[roundedIntensity] ?: ""

        Text(description, fontSize = 14.sp * scale, color = if (selectedEmotion != null) Color.Gray else Color.LightGray)

        Slider(
            value = intensityLevel,
            onValueChange = {
                intensityLevel = it
                if (!sliderTouched) sliderTouched = true
            },
            valueRange = 1f..7f,
            steps = 5,
            modifier = Modifier.fillMaxWidth(),
            enabled = selectedEmotion != null
        )

        Text(
            "Emotion level | Ένταση συναισθήματος",
            fontSize = 16.sp * scale,
            color = if (selectedEmotion != null) Color.Unspecified else Color.LightGray
        )

        Spacer(modifier = Modifier.height(16.dp))

        Button(
            onClick = {
                val timestampExit = System.currentTimeMillis()
                selectedEmotion?.let {
                    val label = if (it.id.toIntOrNull() == 23)
                        customEmotionText.trim().ifEmpty { "Other (no description)" }
                    else it.greekLabel

                    logAudioEmotion(
                        context,
                        username,
                        artworkId,
                        it.id,
                        intensityLevel.toInt(),
                        timestampEntry,
                        timestampExit,
                        label
                    )
                }

                mediaPlayer?.release()
                mediaPlayer = null

                navController.navigate("artworkSelection/$username") {
                    popUpTo("artworkSelection/$username") { inclusive = true }
                }
            },
            enabled = selectedEmotion != null && sliderTouched,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Save & Exit | Αποθήκευση & Έξοδος", fontSize = 16.sp * scale)
        }

        Spacer(modifier = Modifier.height(16.dp))

        Text(
            text = "© 2025 MMAI Team | University of the Aegean",
            color = Color.Gray,
            fontSize = 12.sp * scale,
            textAlign = TextAlign.Center,
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 8.dp)
        )
    }
}

fun formatTime(milliseconds: Int): String {
    val minutes = TimeUnit.MILLISECONDS.toMinutes(milliseconds.toLong())
    val seconds = TimeUnit.MILLISECONDS.toSeconds(milliseconds.toLong()) % 60
    return String.format("%02d:%02d", minutes, seconds)
}
