package com.example.museumemotionapp.screens

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
import com.example.museumemotionapp.models.Emotion
import com.example.museumemotionapp.models.emotions
import com.example.museumemotionapp.utils.logOrUpdateUserEmotion
import com.example.museumemotionapp.LocalFontScale
import kotlin.math.roundToInt

@Composable
fun ArtworkDetailScreen(
    navController: NavController,
    artworkId: String,
    username: String,
    timestampEntry: Long
) {
    val context = LocalContext.current
    val scale = LocalFontScale.current.scale
    var selectedEmotion by remember { mutableStateOf<Emotion?>(null) }
    var intensityLevel by remember { mutableStateOf(4f) }
    var sliderTouched by remember { mutableStateOf(false) }
    var showCustomEmotionDialog by remember { mutableStateOf(false) }
    var customEmotionText by remember { mutableStateOf("") }

    // Disable Android Back Button
    BackHandler { }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text("Artwork ID: $artworkId", style = MaterialTheme.typography.headlineMedium.copy(fontSize = 22.sp * scale))
        Text("User: $username", style = MaterialTheme.typography.bodyMedium.copy(fontSize = 16.sp * scale))

        Spacer(modifier = Modifier.height(16.dp))

        ImageFromAssets(context = context, artworkId = artworkId)

        Spacer(modifier = Modifier.height(16.dp))

        Text("Select how you feel about this artwork:", fontSize = 16.sp * scale)
        Text("Επιλέξτε το πως νιώθετε σχετικά με αυτό το έργο:", fontSize = 16.sp * scale)

        // Emotion Selection
        Box(modifier = Modifier.weight(1f)) {
            LazyColumn {
                items(emotions) { emotion ->
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(8.dp)
                            .clickable {
                                selectedEmotion = emotion
                                sliderTouched = false
                                if (emotion.id.toIntOrNull() == 23){
                                    showCustomEmotionDialog = true
                                    customEmotionText = ""
                                }
                            }
                    ) {
                        RadioButton(
                            selected = selectedEmotion == emotion,
                            onClick = {
                                selectedEmotion = emotion
                                sliderTouched = false
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

        // Dialog for custom emotion input (if 'Other' selected)
        if (showCustomEmotionDialog) {
            AlertDialog(
                onDismissRequest = { showCustomEmotionDialog = false },
                confirmButton = {
                    TextButton(onClick = {
                        showCustomEmotionDialog = false
                    }) {
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

        // Slider description
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

        Text(
            description,
            fontSize = 14.sp * scale,
            color = if (selectedEmotion != null) Color.Gray else Color.LightGray
        )

        // Emotion Level Slider – only active after emotion selection
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
            text = "Emotion level | Ένταση συναισθήματος",
            fontSize = 16.sp * scale,
            color = if (selectedEmotion != null) Color.Unspecified else Color.LightGray
        )

        Spacer(modifier = Modifier.height(16.dp))

        // Save button – only enabled after emotion is selected and slider is moved
        Button(
            onClick = {
                val timestampExit = System.currentTimeMillis()
                selectedEmotion?.let {
                    val finalLabel = if (it.id.toIntOrNull() == 23)
                    customEmotionText.trim().ifEmpty { "Other (no description)" }
                    else it.greekLabel

                    logOrUpdateUserEmotion(
                        context,
                        username,
                        artworkId,
                        it.id,
                        intensityLevel.toInt(),
                        timestampEntry,
                        timestampExit,
                        finalLabel
                    )
                    navController.navigate("audioPlayback/$artworkId/$username")
                }
            },
            enabled = selectedEmotion != null && sliderTouched,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Continue | Συνέχεια", fontSize = 18.sp * scale)
        }

        Spacer(modifier = Modifier.height(16.dp))

        Text(
            text = "© 2025 MMAI Team | University of the Aegean",
            color = Color.Gray,
            textAlign = TextAlign.Center,
            fontSize = 12.sp * scale,
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 8.dp)
        )
    }
}
