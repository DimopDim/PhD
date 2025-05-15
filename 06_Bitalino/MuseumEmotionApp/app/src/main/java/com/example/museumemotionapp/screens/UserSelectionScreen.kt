package com.example.museumemotionapp.screens

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavController
import com.example.museumemotionapp.LocalFontScale

@Composable
fun UserSelectionScreen(navController: NavController) {
    val scale = LocalFontScale.current.scale

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(modifier = Modifier.weight(1f))

        Text(
            "Are you a new or existing user?",
            fontSize = 18.sp * scale,
            textAlign = TextAlign.Center
        )
        Text(
            "Είστε νέος χρήστης ή έχετε λογαριασμό;",
            fontSize = 18.sp * scale,
            textAlign = TextAlign.Center
        )

        Spacer(modifier = Modifier.height(16.dp))

        Button(onClick = { navController.navigate("loginScreen") }) {
            Text("New User / Νέος Χρήστης", fontSize = 16.sp * scale)
        }

        Spacer(modifier = Modifier.height(8.dp))

        Button(onClick = { navController.navigate("existingUserScreen") }) {
            Text("Existing User / Υπάρχων Χρήστης", fontSize = 16.sp * scale)
        }

        Spacer(modifier = Modifier.weight(1f))

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
