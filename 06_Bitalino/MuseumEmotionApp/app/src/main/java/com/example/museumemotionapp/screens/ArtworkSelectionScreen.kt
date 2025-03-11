package com.example.museumemotionapp.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController
import com.example.museumemotionapp.models.artworks

@Composable
fun ArtworkSelectionScreen(username: String, navController: NavController) {
    var searchQuery by remember { mutableStateOf("") } // State for search input

    // Filtered artworks based on search query
    val filteredArtworks = artworks.filter { artwork ->
        artwork.id.contains(searchQuery, ignoreCase = true)
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        Text(
            "Choose an Artwork - Επιλέξτε ένα έργο τέχνης ($username)",
            style = MaterialTheme.typography.headlineMedium
        )

        Spacer(modifier = Modifier.height(16.dp))

        // 🔍 Search Box
        TextField(
            value = searchQuery,
            onValueChange = { searchQuery = it },
            label = { Text("Search by ID") },
            modifier = Modifier.fillMaxWidth()
        )

        Spacer(modifier = Modifier.height(16.dp))

        LazyColumn {
            items(filteredArtworks) { artwork ->
                Button(
                    onClick = {
                        val startTime = System.currentTimeMillis()
                        navController.navigate("artworkDetail/${artwork.id}/$username")
                    },
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("${artwork.id}. ${artwork.title} - ${artwork.greekTitle}")
                }
                Spacer(modifier = Modifier.height(8.dp))
            }
        }
    }
}
