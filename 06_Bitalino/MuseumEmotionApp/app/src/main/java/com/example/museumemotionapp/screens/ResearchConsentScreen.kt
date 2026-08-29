package com.example.museumemotionapp.screens

import androidx.compose.runtime.Composable
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.Alignment
import androidx.navigation.NavController
import com.example.museumemotionapp.LocalFontScale

@Composable
fun ResearchConsentScreen(navController: NavController, username: String) {
    val scale = LocalFontScale.current.scale

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Κεντρικό περιεχόμενο
        Spacer(modifier = Modifier.weight(1f))

        Text(
            text = "Συμμετοχή στην Έρευνα",
            fontSize = 22.sp * scale,
            textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth()
        )

        Spacer(modifier = Modifier.height(24.dp))

        Button(
            onClick = {
                navController.navigate("researchInfo/$username")
            },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("📄 Πληροφόριση Συμμετοχόντων", fontSize = 16.sp * scale)
        }

        Spacer(modifier = Modifier.height(16.dp))

        Button(
            onClick = {
                navController.navigate("consentFormScreen/$username")
            },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Φόρμα Συναίνεσης", fontSize = 16.sp * scale)
        }

        Spacer(modifier = Modifier.height(16.dp))

        Button(
            onClick = {
                navController.popBackStack("userSelection", inclusive = false)
            },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Δεν επιθυμώ να συμμετάσχω", fontSize = 16.sp * scale)
        }

        Spacer(modifier = Modifier.weight(1f))

        // Footer
        Text(
            text = "© 2025 MMAI Team | University of the Aegean",
            color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
            textAlign = TextAlign.Center,
            fontSize = 12.sp * scale,
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 8.dp)
        )
    }
}
