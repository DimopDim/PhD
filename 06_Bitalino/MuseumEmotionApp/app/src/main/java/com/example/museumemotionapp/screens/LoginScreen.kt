package com.example.museumemotionapp.screens

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.os.Environment
import android.util.Log
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.navigation.NavController
import com.example.museumemotionapp.LocalFontScale
import java.io.File

@Composable
fun LoginScreen(navController: NavController) {
    val context = LocalContext.current
    val activity = context as? Activity
    val scale = LocalFontScale.current.scale

    var username by remember { mutableStateOf("") }
    var showErrorDialog by remember { mutableStateOf(false) }
    var showSuccessDialog by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf("") }

    fun requestStoragePermission(): Boolean {
        val permission = Manifest.permission.WRITE_EXTERNAL_STORAGE
        val granted = ContextCompat.checkSelfPermission(context, permission) == PackageManager.PERMISSION_GRANTED
        if (!granted && activity != null) {
            ActivityCompat.requestPermissions(activity, arrayOf(permission), 1001)
        }
        return granted
    }

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
                    if (requestStoragePermission()) {
                        try {
                            val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                            val museumEmotionFolder = File(downloadsDir, "MuseumEmotion")
                            val userFolder = File(museumEmotionFolder, username)

                            Log.d("LoginScreen", "Creating path: ${userFolder.absolutePath}")

                            if (userFolder.exists()) {
                                errorMessage = "Username already exists in Downloads folder."
                                showErrorDialog = true
                            } else {
                                val created = userFolder.mkdirs()
                                if (created) {
                                    showSuccessDialog = true
                                } else {
                                    errorMessage = "Failed to create folder: ${userFolder.absolutePath}"
                                    showErrorDialog = true
                                }
                            }
                        } catch (e: Exception) {
                            errorMessage = "Exception: ${e.localizedMessage}"
                            showErrorDialog = true
                        }
                    } else {
                        errorMessage = "Storage permission not granted. Please allow it and try again."
                        showErrorDialog = true
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
                    Text("Error / Σφάλμα", fontSize = 18.sp * scale)
                },
                text = {
                    Text(errorMessage, fontSize = 14.sp * scale)
                }
            )
        }

        if (showSuccessDialog) {
            AlertDialog(
                onDismissRequest = { showSuccessDialog = false },
                confirmButton = {
                    Button(onClick = {
                        showSuccessDialog = false
                        navController.navigate("researchConsent/$username")
                    }) {
                        Text("OK", fontSize = 16.sp * scale)
                    }
                },
                title = {
                    Text("User Created | Ο χρήστης δημιουργήθηκε", fontSize = 18.sp * scale)
                },
                text = {
                    Text("Your account has been successfully created. | Επιτυχής δημιουργία λογαριασμού", fontSize = 14.sp * scale)
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
