package com.example.museumemotionapp.screens

import com.example.museumemotionapp.utils.getVisitedArtworksFromLog
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
import androidx.compose.ui.unit.sp
import com.example.museumemotionapp.LocalFontScale

@Composable
fun ArtworkListScreen(navController: NavController, username: String) {
    val scale = LocalFontScale.current.scale
    var searchText by remember { mutableStateOf("") }
    var selectedArtwork by remember { mutableStateOf<Artwork?>(null) }
    val context = LocalContext.current

    val visitedArtworks = remember { mutableStateOf(setOf<String>()) }

    LaunchedEffect(username) {
        visitedArtworks.value = getVisitedArtworksFromLog(context, username)
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            "Select an Artwork",
            fontSize = 24.sp * scale,
            textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth()
        )
        Text(
            "Επιλέξτε ένα έργο τέχνης",
            fontSize = 24.sp * scale,
            textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth()
        )
        Text(
            "User: $username",
            fontSize = 20.sp * scale,
            textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth()
        )

        Spacer(modifier = Modifier.height(8.dp))

        // Search Box
        OutlinedTextField(
            value = searchText,
            onValueChange = { searchText = it },
            label = {
                Text("Search by ID / Αναζήτηση κατά ID", fontSize = 14.sp * scale)
            },
            keyboardOptions = KeyboardOptions.Default.copy(keyboardType = KeyboardType.Number),
            modifier = Modifier
                .fillMaxWidth()
                .padding(8.dp)
        )

        Spacer(modifier = Modifier.height(8.dp))

        // Artwork List
        Box(modifier = Modifier.weight(1f)) {
            LazyColumn {
                items(artworks.filter { it.id.contains(searchText, ignoreCase = true) }) { artwork ->
                    val isVisited = visitedArtworks.value.contains(artwork.id)

                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(8.dp)
                            .clickable { selectedArtwork = artwork },
                        colors = CardDefaults.cardColors(
                            containerColor = if (isVisited) Color.LightGray else Color.White
                        )
                    ) {
                        Row(
                            modifier = Modifier.padding(16.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = "${artwork.id}: ${artwork.title} | ${artwork.greekTitle}",
                                fontSize = 16.sp * scale,
                                color = if (isVisited) Color.Gray else Color.Black
                            )
                        }
                    }
                }
            }
        }

        // Footer
        Text(
            text = "© 2025 MMAI Team | University of the Aegean",
            color = Color.Gray,
            textAlign = TextAlign.Center,
            fontSize = 12.sp * scale,
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 8.dp)
        )

        // Confirmation Dialog
        selectedArtwork?.let { artwork ->
            AlertDialog(
                onDismissRequest = { selectedArtwork = null },
                title = {
                    Text("Confirm | Επιβεβαιώστε", fontSize = 18.sp * scale)
                },
                text = {
                    Column(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Text("Is this the correct artwork?", fontSize = 16.sp * scale)
                        Text("Είναι αυτό το σωστό έργο?", fontSize = 16.sp * scale)
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
                            logOrUpdateUserEmotion(context, username, artwork.id, null, null, timestampEntry, null)
                            navController.navigate("artworkDetail/${artwork.id}/$username/$timestampEntry")
                            selectedArtwork = null
                        }) {
                            Text("Yes / Ναι", fontSize = 16.sp * scale)
                        }

                        Spacer(modifier = Modifier.width(16.dp))

                        Button(onClick = { selectedArtwork = null }) {
                            Text("No / Όχι", fontSize = 16.sp * scale)
                        }
                    }
                }
            )
        }
    }
}
