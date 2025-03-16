package com.example.museumemotionapp.screens

import android.os.Environment
import androidx.compose.foundation.layout.*
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
fun LoginScreen(navController: NavController) {
    var username by remember { mutableStateOf("") }
    var showErrorDialog by remember { mutableStateOf(false) }
    var showSuccessDialog by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // **Wrap the Main Content in a Centered Column**
        Column(
            modifier = Modifier.weight(1f),
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
                        showErrorDialog = true
                    } else {
                        userFolder.mkdirs()
                        showSuccessDialog = true
                    }
                }
            }) {
                Text("Continue / Επόμενο")
            }
        }

        // **Error Dialog if User Exists**
        if (showErrorDialog) {
            AlertDialog(
                onDismissRequest = { showErrorDialog = false },
                confirmButton = {
                    Button(onClick = { showErrorDialog = false }) {
                        Text("OK")
                    }
                },
                title = { Text("Username Exists | Το όνομα χρήστη υπάρχει") },
                text = { Text("This username already exists. Please enter a different name. | Το όνομα χρήστη υπάρχει ήδη. Παρακαλώ δοκιμάστε ένα διαφορετικό όνομα.") }
            )
        }

        // **Success Dialog if New User Created**
        if (showSuccessDialog) {
            AlertDialog(
                onDismissRequest = { showSuccessDialog = false },
                confirmButton = {
                    Button(onClick = {
                        showSuccessDialog = false
                        navController.navigate("artworkSelection/$username") // Navigate next
                    }) {
                        Text("OK")
                    }
                },
                title = { Text("User Created | Ο χρήστης δημιουργήθηκε") },
                text = { Text("Your account has been successfully created. | Επιτυχής δημιουργία λογαριασμού") }
            )
        }

        Spacer(modifier = Modifier.height(16.dp))

        // **Footer (Copyright Text)**
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
