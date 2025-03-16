package com.example.museumemotionapp.screens

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController

@Composable
fun UserSelectionScreen(navController: NavController) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(modifier = Modifier.weight(1f)) // Push content to center

        Text("Are you a new or existing user?")
        Text("Είστε νέος χρήστης ή έχετε λογαριασμό;")

        Spacer(modifier = Modifier.height(16.dp))

        Button(onClick = { navController.navigate("loginScreen") }) {
            Text("New User / Νέος Χρήστης")
        }

        Spacer(modifier = Modifier.height(8.dp))

        Button(onClick = { navController.navigate("existingUserScreen") }) {
            Text("Existing User / Υπάρχων Χρήστης")
        }

        Spacer(modifier = Modifier.weight(1f)) // Push footer to bottom

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
