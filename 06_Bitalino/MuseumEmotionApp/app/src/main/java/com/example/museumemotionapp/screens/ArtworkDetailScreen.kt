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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController
import com.example.museumemotionapp.models.Emotion
import com.example.museumemotionapp.models.emotions
import com.example.museumemotionapp.utils.logOrUpdateUserEmotion
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.sp
import com.example.museumemotionapp.LocalFontScale
import kotlin.math.roundToInt

@Composable
fun ArtworkDetailScreen(navController: NavController, artworkId: String, username: String, timestampEntry: Long) {
    val context = LocalContext.current
    val scale = LocalFontScale.current.scale
    var selectedEmotion by remember { mutableStateOf<Emotion?>(null) }
    var intensityLevel by remember { mutableStateOf(4f) }

    // **Disable Android Back Button**
    BackHandler {
        // Blocks back navigation
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "Artwork ID: $artworkId",
            style = MaterialTheme.typography.headlineMedium.copy(fontSize = 22.sp * scale)
        )
        Text(
            text = "User: $username",
            style = MaterialTheme.typography.bodyMedium.copy(fontSize = 16.sp * scale)
        )

        Spacer(modifier = Modifier.height(16.dp))

        // Display Artwork Image
        ImageFromAssets(context = context, artworkId = artworkId)

        Spacer(modifier = Modifier.height(16.dp))

        Text(text = "Select how you feel about this artwork:", fontSize = 16.sp * scale)
        Text(text = "Επιλέξτε το πως νιώθετε σχετικά με αυτό το έργο:", fontSize = 16.sp * scale)

        // Emotion Selection List
        Box(
            modifier = Modifier.weight(1f) // Ensures list is scrollable, button remains fixed
        ) {
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
                            Text(text = "${emotion.id}. ${emotion.englishLabel}", fontSize = 16.sp * scale)
                            Text(
                                text = emotion.greekLabel,
                                style = MaterialTheme.typography.bodySmall.copy(fontSize = 14.sp * scale)
                            )
                        }
                    }
                }
            }
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
        Text(description, fontSize = 14.sp * scale, color = Color.Gray)
        Slider(
            value = intensityLevel,
            onValueChange = { intensityLevel = it },
            valueRange = 1f..7f,
            steps = 5,
            modifier = Modifier.fillMaxWidth()
        )
        Text(text = "Emotion level | Ένταση  συναισθήματος", fontSize = 16.sp * scale)
        Spacer(modifier = Modifier.height(16.dp))

        Button(
            onClick = {
                val timestampExit = System.currentTimeMillis()
                selectedEmotion?.let {
                    logOrUpdateUserEmotion(
                        context,
                        username,
                        artworkId,
                        it.id,
                        intensityLevel.toInt(),
                        timestampEntry,
                        timestampExit
                    )
                    navController.navigate("audioPlayback/$artworkId/$username")
                }
            },
            enabled = selectedEmotion != null,
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
