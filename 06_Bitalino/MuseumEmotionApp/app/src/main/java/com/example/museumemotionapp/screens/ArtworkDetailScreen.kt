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


@Composable
fun ArtworkDetailScreen(navController: NavController, artworkId: String, username: String, timestampEntry: Long) {
    val context = LocalContext.current
    var selectedEmotion by remember { mutableStateOf<Emotion?>(null) }

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
        Text(text = "Artwork ID: $artworkId", style = MaterialTheme.typography.headlineMedium)
        Text(text = "User: $username", style = MaterialTheme.typography.bodyMedium)

        Spacer(modifier = Modifier.height(16.dp))

        // Display Artwork Image
        ImageFromAssets(context = context, artworkId = artworkId)

        Spacer(modifier = Modifier.height(16.dp))

        Text(text = "Select how you feel about this artwork:")
        Text(text = "Επιλέξτε το πως νιώθετε σχετικά με αυτό το έργο:")

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
                            Text(text = "${emotion.id}. ${emotion.englishLabel}")
                            Text(text = emotion.greekLabel, style = MaterialTheme.typography.bodySmall)
                        }
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // "Continue" Button - Always Visible
        Button(
            onClick = {
                val timestampExit = System.currentTimeMillis()
                selectedEmotion?.let {
                    logOrUpdateUserEmotion(context, username, artworkId, it.id, timestampEntry, timestampExit)
                    navController.navigate("audioPlayback/$artworkId/$username")
                }
            },
            enabled = selectedEmotion != null,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Continue | Συνέχεια")
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
