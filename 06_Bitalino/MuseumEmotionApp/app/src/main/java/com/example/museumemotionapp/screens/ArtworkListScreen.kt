package com.example.museumemotionapp.screens

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.graphics.Color
import androidx.navigation.NavController
import com.example.museumemotionapp.models.Artwork
import com.example.museumemotionapp.models.artworks
import com.example.museumemotionapp.utils.logOrUpdateUserEmotion
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.unit.sp

@Composable
fun ArtworkListScreen(navController: NavController, username: String) {
    var searchText by remember { mutableStateOf("") }
    var selectedArtwork by remember { mutableStateOf<Artwork?>(null) }
    val context = LocalContext.current

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            "Select an Artwork",
            style = TextStyle(fontSize = 24.sp),
            textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth()
        )
        Text(
            "Επιλέξτε ένα έργο τέχνης",
            style = TextStyle(fontSize = 24.sp),
            textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth()
        )
        Text(
            "User: $username",
            style = TextStyle(fontSize = 20.sp),
            textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth()
        )

        Spacer(modifier = Modifier.height(8.dp))

        // Search Box with Frame and Numeric Keyboard
        OutlinedTextField(
            value = searchText,
            onValueChange = { searchText = it },
            label = { Text("Search by ID / Αναζήτηση κατά ID") },
            keyboardOptions = KeyboardOptions.Default.copy(keyboardType = KeyboardType.Number), // Numeric Keyboard Only
            modifier = Modifier
                .fillMaxWidth()
                .padding(8.dp)
        )

        Spacer(modifier = Modifier.height(8.dp))

        // Artwork List (Scrollable)
        Box(modifier = Modifier.weight(1f)) {
            LazyColumn {
                items(artworks.filter { it.id.contains(searchText, ignoreCase = true) }) { artwork ->
                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(8.dp)
                            .clickable { selectedArtwork = artwork }
                    ) {
                        Row(
                            modifier = Modifier.padding(16.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(text = "${artwork.id}: ${artwork.title} | ${artwork.greekTitle}")
                        }
                    }
                }
            }
        }

        // Footer (Copyright Text)
        Text(
            text = "© 2025 MMAI Team | University of the Aegean",
            color = Color.Gray,
            textAlign = TextAlign.Center,
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 8.dp)
        )

        // Popup Dialog for Artwork Confirmation (with image restored)
        selectedArtwork?.let { artwork ->
            AlertDialog(
                onDismissRequest = { selectedArtwork = null },
                title = {
                    Text(text = "Confirm | Επιβεβαιώστε ")

                },
                text = {
                    Column(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Text("Is this the correct artwork?")
                        Text("Είναι αυτό το σωστό έργο?")
                        Spacer(modifier = Modifier.height(8.dp))
                        ImageFromAssets(context, artwork.id)
                    }
                },
                confirmButton = {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.Center
                    ) {
                        Button(onClick = {
                            val timestampEntry = System.currentTimeMillis()

                            // Log user entering the artwork selection
                            logOrUpdateUserEmotion(context, username, artwork.id, null, timestampEntry, null)

                            navController.navigate("artworkDetail/${artwork.id}/$username/$timestampEntry")
                            selectedArtwork = null
                        }) {
                            Text("Yes / Ναι")
                        }

                        Spacer(modifier = Modifier.width(16.dp))

                        Button(onClick = { selectedArtwork = null }) {
                            Text("No / Όχι")
                        }
                    }
                }
            )
        }
    }
}
