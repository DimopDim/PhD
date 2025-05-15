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
import androidx.compose.ui.unit.sp
import androidx.navigation.NavController
import com.example.museumemotionapp.LocalFontScale
import java.io.File

@Composable
fun LoginScreen(navController: NavController) {
    val scale = LocalFontScale.current.scale
    var username by remember { mutableStateOf("") }
    var showErrorDialog by remember { mutableStateOf(false) }
    var showSuccessDialog by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Column(
            modifier = Modifier.weight(1f),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Button(onClick = { navController.popBackStack() }) {
                Text("Back / Πίσω", fontSize = 16.sp * scale)
            }

            Spacer(modifier = Modifier.height(16.dp))

            Text("Enter Your Name / Εισάγετε το όνομα σας", fontSize = 18.sp * scale)

            TextField(
                value = username,
                onValueChange = { username = it },
                label = { Text("Username / Όνομα χρήστη", fontSize = 14.sp * scale) },
                textStyle = LocalTextStyle.current.copy(fontSize = 16.sp * scale),
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
                Text("Continue / Επόμενο", fontSize = 16.sp * scale)
            }
        }

        if (showErrorDialog) {
            AlertDialog(
                onDismissRequest = { showErrorDialog = false },
                confirmButton = {
                    Button(onClick = { showErrorDialog = false }) {
                        Text("OK", fontSize = 16.sp * scale)
                    }
                },
                title = {
                    Text("Username Exists | Το όνομα χρήστη υπάρχει", fontSize = 18.sp * scale)
                },
                text = {
                    Text(
                        "This username already exists. Please enter a different name. | Το όνομα χρήστη υπάρχει ήδη. Παρακαλώ δοκιμάστε ένα διαφορετικό όνομα.",
                        fontSize = 14.sp * scale
                    )
                }
            )
        }

        if (showSuccessDialog) {
            AlertDialog(
                onDismissRequest = { showSuccessDialog = false },
                confirmButton = {
                    Button(onClick = {
                        showSuccessDialog = false
                        navController.navigate("artworkSelection/$username")
                    }) {
                        Text("OK", fontSize = 16.sp * scale)
                    }
                },
                title = {
                    Text("User Created | Ο χρήστης δημιουργήθηκε", fontSize = 18.sp * scale)
                },
                text = {
                    Text("Your account has been successfully created. | Επιτυχής δημιουργία λογαριασμού",
                        fontSize = 14.sp * scale
                    )
                }
            )
        }

        Spacer(modifier = Modifier.height(16.dp))

        Text(
            text = "© 2025 MMAI Team | University of the Aegean",
            color = Color.Gray,
            textAlign = TextAlign.Center,
            fontSize = 12.sp * scale,
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 8.dp)
        )
    }
}
