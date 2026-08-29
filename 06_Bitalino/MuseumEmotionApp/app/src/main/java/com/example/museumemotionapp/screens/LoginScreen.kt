@file:OptIn(androidx.compose.foundation.ExperimentalFoundationApi::class)

package com.example.museumemotionapp.screens

import android.util.Log
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.relocation.BringIntoViewRequester
import androidx.compose.foundation.relocation.bringIntoViewRequester
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
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
import androidx.navigation.NavController
import com.example.museumemotionapp.LocalFontScale
import com.example.museumemotionapp.utils.getUserFolder   // ✅ σωστό helper
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

@Composable
fun LoginScreen(
    navController: NavController,
    onUsernameConfirmed: (String) -> Unit
) {
    val context = LocalContext.current
    val scale = LocalFontScale.current.scale
    val focusManager = LocalFocusManager.current
    val coroutineScope = rememberCoroutineScope()

    var username by remember { mutableStateOf("") }
    var showErrorDialog by remember { mutableStateOf(false) }
    var showSuccessDialog by remember { mutableStateOf(false) }
    var errorMessage by remember { mutableStateOf("") }

    val bringIntoViewRequester = remember { BringIntoViewRequester() }
    val focusRequester = remember { FocusRequester() }

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
                Text(
                    "Enter Your Name / Εισάγετε το όνομα σας",
                    fontSize = 18.sp * scale
                )

                Spacer(modifier = Modifier.height(8.dp))

                TextField(
                    value = username,
                    onValueChange = { username = it },
                    label = { Text("Username / Όνομα χρήστη", fontSize = 14.sp * scale) },
                    textStyle = TextStyle(fontSize = 16.sp * scale),
                    modifier = Modifier
                        .fillMaxWidth()
                        .focusRequester(focusRequester)
                        .onFocusChanged { focusState ->
                            if (focusState.isFocused) {
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
                            val trimmedUsername = username.trim()
                            username = trimmedUsername

                            try {
                                val userFolder = getUserFolder(context, trimmedUsername)
                                Log.d("LoginScreen", "User folder path: ${userFolder.absolutePath}")

                                if (userFolder.exists()) {
                                    // Username already exists
                                    errorMessage =
                                        "Username already exists.\n\nΤο όνομα χρήστη υπάρχει ήδη."
                                    showErrorDialog = true
                                } else {
                                    val created = userFolder.mkdirs()
                                    if (created) {
                                        showSuccessDialog = true
                                    } else {
                                        errorMessage =
                                            "Failed to create folder:\n${userFolder.absolutePath}"
                                        showErrorDialog = true
                                    }
                                }
                            } catch (e: Exception) {
                                errorMessage =
                                    "Exception: ${e.localizedMessage ?: "Unknown error"}"
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
                    text = {
                        Text(
                            "Your account has been successfully created.\n\nΕπιτυχής δημιουργία λογαριασμού",
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
}
