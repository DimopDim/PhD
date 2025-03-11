package com.example.museumemotionapp.screens

import android.os.Environment
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController
import com.example.museumemotionapp.utils.FileUtils.logUserLogin
import java.io.File

@Composable
fun LoginScreen(navController: NavController) {
    var username by remember { mutableStateOf("") }
    var showErrorDialog by remember { mutableStateOf(false) } // Error state
    var showSuccessDialog by remember { mutableStateOf(false) } // Success state

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Button(onClick = { navController.popBackStack() }) {
            Text("Back / Πίσω")
        }

        Spacer(modifier = Modifier.height(16.dp))

        Text("Enter Your Name / Εισάγετε το όνομα σας")

        TextField(
            value = username,
            onValueChange = { username = it },
            label = { Text("Username / Όνομα χρήστη") },
            modifier = Modifier.fillMaxWidth()
        )

        Spacer(modifier = Modifier.height(16.dp))

        Button(onClick = {
            if (username.isNotBlank()) {
                val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                val museumEmotionFolder = File(downloadsDir, "MuseumEmotion")
                val userFolder = File(museumEmotionFolder, username)

                if (userFolder.exists()) {
                    showErrorDialog = true  // Show error message when user already exists
                } else {
                    userFolder.mkdirs()  // Create new user folder
                    logUserLogin(username)  // Log user login
                    showSuccessDialog = true
                }
            }
        }) {
            Text("Continue / Επόμενο")
        }

        // Error Dialog for Existing User
        if (showErrorDialog) {
            AlertDialog(
                onDismissRequest = { showErrorDialog = false },
                confirmButton = {
                    Button(onClick = { showErrorDialog = false }) {
                        Text("OK")
                    }
                },
                title = { Text("Username Exists") },
                text = { Text("Be careful, this username already exists. Change it before proceeding or search it in the 'Existing Users'.") }
            )
        }

        // Success Dialog for New User
        if (showSuccessDialog) {
            AlertDialog(
                onDismissRequest = { showSuccessDialog = false },
                confirmButton = {
                    Button(onClick = {
                        showSuccessDialog = false
                        navController.navigate("artworkSelection/$username")
                    }) {
                        Text("OK")
                    }
                },
                title = { Text("User Created") },
                text = { Text("Your account has been successfully created.") }
            )
        }
    }
}
