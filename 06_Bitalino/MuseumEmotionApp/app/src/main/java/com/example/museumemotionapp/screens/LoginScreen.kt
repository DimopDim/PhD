@file:OptIn(androidx.compose.foundation.ExperimentalFoundationApi::class)

package com.example.museumemotionapp.screens

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.os.Environment
import android.util.Log
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.relocation.BringIntoViewRequester
import androidx.compose.foundation.relocation.bringIntoViewRequester
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.navigation.NavController
import com.example.museumemotionapp.LocalFontScale
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.io.File

@Composable
fun LoginScreen(
    navController: NavController,
    onUsernameConfirmed: (String) -> Unit
) {
    val context = LocalContext.current
    val activity = context as? Activity
    val scale = LocalFontScale.current.scale
    val focusManager = LocalFocusManager.current
    val coroutineScope = rememberCoroutineScope()

    var username by remember { mutableStateOf("") }
    var showErrorDialog by remember { mutableStateOf(false) }
    var showSuccessDialog by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf("") }

    val bringIntoViewRequester = remember { BringIntoViewRequester() }
    val focusRequester = remember { FocusRequester() }

    fun requestStoragePermission(): Boolean {
        val permission = Manifest.permission.WRITE_EXTERNAL_STORAGE
        val granted = ContextCompat.checkSelfPermission(context, permission) == PackageManager.PERMISSION_GRANTED
        if (!granted && activity != null) {
            ActivityCompat.requestPermissions(activity, arrayOf(permission), 1001)
        }
        return granted
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .imePadding()
            .clickable(
                interactionSource = remember { MutableInteractionSource() },
                indication = null
            ) {
                focusManager.clearFocus()
            }
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp)
                .verticalScroll(rememberScrollState())
                .bringIntoViewRequester(bringIntoViewRequester),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Column(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth(),
                verticalArrangement = Arrangement.Center,
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text("Enter Your Name / Εισάγετε το όνομα σας", fontSize = 18.sp * scale)

                Spacer(modifier = Modifier.height(8.dp))

                TextField(
                    value = username,
                    onValueChange = { username = it },
                    label = { Text("Username / Όνομα χρήστη", fontSize = 14.sp * scale) },
                    textStyle = TextStyle(fontSize = 16.sp * scale),
                    modifier = Modifier
                        .fillMaxWidth()
                        .focusRequester(focusRequester)
                        .onFocusChanged {
                            if (it.isFocused) {
                                coroutineScope.launch {
                                    delay(300)
                                    bringIntoViewRequester.bringIntoView()
                                }
                            }
                        }
                )

                Spacer(modifier = Modifier.height(16.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Button(onClick = {
                        focusManager.clearFocus()
                        navController.popBackStack()
                    }) {
                        Text("Back / Πίσω", fontSize = 16.sp * scale)
                    }

                    Button(onClick = {
                        focusManager.clearFocus()
                        if (username.isNotBlank()) {
                            if (requestStoragePermission()) {
                                try {
                                    val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                                    val museumEmotionFolder = File(downloadsDir, "MuseumEmotion")
                                    val userFolder = File(museumEmotionFolder, username)

                                    Log.d("LoginScreen", "Creating path: ${userFolder.absolutePath}")

                                    if (userFolder.exists()) {
                                        errorMessage = "Username already exists.\n\nΤο όνομα χρήστη υπάρχει ήδη."
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
                        Text("Continue | Επόμενο", fontSize = 16.sp * scale)
                    }
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
                    title = { Text("Error | Σφάλμα", fontSize = 18.sp * scale) },
                    text = { Text(errorMessage, fontSize = 14.sp * scale) }
                )
            }

            if (showSuccessDialog) {
                AlertDialog(
                    onDismissRequest = { showSuccessDialog = false },
                    confirmButton = {
                        Button(onClick = {
                            showSuccessDialog = false
                            onUsernameConfirmed(username)
                        }) {
                            Text("OK", fontSize = 16.sp * scale)
                        }
                    },
                    title = { Text("User Created | Ο χρήστης δημιουργήθηκε", fontSize = 18.sp * scale) },
                    text = { Text("Your account has been successfully created.\n\nΕπιτυχής δημιουργία λογαριασμού", fontSize = 14.sp * scale) }
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
}
