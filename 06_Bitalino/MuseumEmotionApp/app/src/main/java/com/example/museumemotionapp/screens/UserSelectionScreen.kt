package com.example.museumemotionapp.screens

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController

@Composable
fun UserSelectionScreen(navController: NavController) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text("Are you a new or existing user?")
        Text("Είστε νέος χρήστης ή έχετε λογαριασμό;")

        Spacer(modifier = Modifier.height(16.dp))

        Button(onClick = { navController.navigate("newUser") }) {
            Text("New User / Νέος Χρήστης")
        }

        Spacer(modifier = Modifier.height(8.dp))

        Button(onClick = { navController.navigate("existingUser") }) {
            Text("Existing User / Υπάρχων Χρήστης")
        }
    }
}