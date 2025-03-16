package com.example.museumemotionapp.screens

import android.os.Environment
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController
import java.io.File

@Composable
fun ExistingUserScreen(navController: NavController) {
    val users = getExistingUsers()  // Get list of existing users

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Content wrapped in a centered Column
        Column(
            modifier = Modifier.weight(1f), // Pushes content to the middle of the screen
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Back Button
            Button(onClick = { navController.popBackStack() }) {
                Text("Back / Πίσω")
            }

            Spacer(modifier = Modifier.height(16.dp))

            Text("Select an existing user")
            Text("Επιλέξτε έναν υπάρχοντα χρήστη")

            Spacer(modifier = Modifier.height(16.dp))

            if (users.isEmpty()) {
                Text("No existing users found!")
                Text("Δεν βρέθηκαν χρήστες!")
            } else {
                LazyColumn(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(300.dp) // Set a fixed height to keep it centered
                ) {
                    items(users) { username ->
                        Card(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(8.dp)
                                .clickable {
                                    navController.navigate("artworkSelection/$username")
                                }
                        ) {
                            Text(
                                text = username,
                                modifier = Modifier.padding(16.dp)
                            )
                        }
                    }
                }
            }
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

// Function to get existing users from "Download/MuseumEmotion/"
fun getExistingUsers(): List<String> {
    val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
    val museumEmotionFolder = File(downloadsDir, "MuseumEmotion")

    return if (museumEmotionFolder.exists()) {
        museumEmotionFolder.list()?.toList() ?: emptyList()
    } else {
        emptyList()
    }
}
