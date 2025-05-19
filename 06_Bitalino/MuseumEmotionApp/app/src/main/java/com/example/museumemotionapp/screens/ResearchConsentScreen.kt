package com.example.museumemotionapp.screens

import androidx.compose.runtime.Composable
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.ui.Modifier
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
        Text("Συμμετοχή στην Έρευνα", fontSize = 22.sp * scale)

        Spacer(modifier = Modifier.height(16.dp))

        Button(onClick = {
            navController.navigate("researchInfo/$username")
        }) {
            Text("📄 Πληροφόριση Συμμετοχόντων", fontSize = 16.sp * scale)
        }

        Spacer(modifier = Modifier.height(16.dp))

        Button(onClick = {
            navController.navigate("consentFormScreen/$username")
        }) {
            Text("Φόρμα Συναίνεσης", fontSize = 16.sp * scale)
        }

        Spacer(modifier = Modifier.height(16.dp))

        Button(onClick = {
            navController.popBackStack("userSelection", inclusive = false)
        }) {
            Text("Δεν επιθυμώ να συμμετάσχω", fontSize = 16.sp * scale)
        }
    }
}
