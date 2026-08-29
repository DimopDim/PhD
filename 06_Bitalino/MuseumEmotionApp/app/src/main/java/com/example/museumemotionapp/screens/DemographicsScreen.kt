@file:OptIn(androidx.compose.foundation.ExperimentalFoundationApi::class)

package com.example.museumemotionapp.screens

import android.app.Activity
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.relocation.BringIntoViewRequester
import androidx.compose.foundation.relocation.bringIntoViewRequester
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue        // ✅ ΑΥΤΑ ΕΔΩ
import androidx.compose.runtime.setValue        // ✅
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Modifier
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavController
import com.example.museumemotionapp.LocalFontScale
import com.example.museumemotionapp.models.demographicsQuestions
import com.example.museumemotionapp.models.genderOptions
import com.example.museumemotionapp.models.handednessOptions
import com.example.museumemotionapp.utils.getUserFolder
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.io.File

@Composable
fun DemographicsScreen(navController: NavController, username: String) {
    val context = LocalContext.current
    val activity = context as? Activity
    val scale = LocalFontScale.current.scale
    val scrollState = rememberScrollState()
    val focusManager = LocalFocusManager.current
    val coroutineScope = rememberCoroutineScope()
    val bringIntoViewRequester = remember { BringIntoViewRequester() }

    val answers = remember {
        mutableStateListOf<String?>(*(Array(demographicsQuestions.size) { null }))
    }

    val ageOptions = listOf("Προτιμώ να μην την αναφέρω") + (18..80).map { it.toString() }
    var ageDropdownExpanded by remember { mutableStateOf(false) }   // ✅ Boolean state με delegation

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
                .verticalScroll(scrollState)
                .bringIntoViewRequester(bringIntoViewRequester),
        ) {
            Text(
                "Παρακαλώ απαντήστε στις παρακάτω ερωτήσεις.",
                fontSize = 18.sp * scale
            )
            Spacer(modifier = Modifier.height(16.dp))

            demographicsQuestions.forEachIndexed { index, question ->
                Text(
                    text = "${index + 1}. $question",
                    fontSize = 14.sp * scale
                )

                when (index) {
                    // 1. Φύλο
                    0 -> {
                        genderOptions.forEach { option ->
                            Row(
                                verticalAlignment = androidx.compose.ui.Alignment.CenterVertically
                            ) {
                                RadioButton(
                                    selected = answers[index] == option,
                                    onClick = { answers[index] = option }
                                )
                                Text(option, fontSize = 14.sp * scale)
                            }
                        }
                    }

                    // 2. Ηλικία (dropdown)
                    1 -> {
                        Box {
                            OutlinedTextField(
                                value = answers[index] ?: "",
                                onValueChange = {},
                                modifier = Modifier.fillMaxWidth(),
                                readOnly = true,
                                label = { Text("Επιλέξτε ηλικία", fontSize = 14.sp * scale) },
                                trailingIcon = {
                                    androidx.compose.material3.IconButton(
                                        onClick = { ageDropdownExpanded = !ageDropdownExpanded } // ✅ ! σε Boolean
                                    ) {
                                        androidx.compose.material3.Icon(
                                            Icons.Default.ArrowDropDown,
                                            contentDescription = "Dropdown"
                                        )
                                    }
                                }
                            )
                            DropdownMenu(
                                expanded = ageDropdownExpanded,
                                onDismissRequest = { ageDropdownExpanded = false }
                            ) {
                                ageOptions.forEach { age ->
                                    DropdownMenuItem(
                                        text = { Text(age) },
                                        onClick = {
                                            answers[index] = age
                                            ageDropdownExpanded = false
                                        }
                                    )
                                }
                            }
                        }
                    }

                    // 8. Χειροπλαγία
                    7 -> {
                        handednessOptions.forEach { option ->
                            Row(
                                verticalAlignment = androidx.compose.ui.Alignment.CenterVertically
                            ) {
                                RadioButton(
                                    selected = answers[index] == option,
                                    onClick = { answers[index] = option }
                                )
                                Text(option, fontSize = 14.sp * scale)
                            }
                        }
                    }

                    // Όλα τα υπόλοιπα: ελεύθερο κείμενο
                    else -> {
                        val focusRequester = remember { FocusRequester() }

                        OutlinedTextField(
                            value = answers[index] ?: "",
                            onValueChange = { answers[index] = it },
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
                    }
                }

                Spacer(modifier = Modifier.height(16.dp))
            }

            Button(
                onClick = {
                    val textOutput = demographicsQuestions.mapIndexed { i, q ->
                        "${i + 1}. $q\nΑπάντηση: ${answers[i] ?: "(καμία απάντηση)"}\n"
                    }.joinToString("\n")

                    try {
                        // SAFE: app-specific external storage
                        val userDir: File = getUserFolder(context, username)
                        if (!userDir.exists()) userDir.mkdirs()

                        val outputFile = File(userDir, "demographics.txt")
                        outputFile.writeText(textOutput)
                    } catch (e: Exception) {
                        e.printStackTrace()
                    }

                    navController.navigate("panasBegin/$username")
                }
            ) {
                Text("Συνέχεια", fontSize = 16.sp * scale)
            }

            Spacer(modifier = Modifier.height(24.dp))

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
