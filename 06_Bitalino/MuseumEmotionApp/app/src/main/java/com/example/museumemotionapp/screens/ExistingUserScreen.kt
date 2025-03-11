package com.example.museumemotionapp.screens

import android.os.Environment
import androidx.compose.foundation.layout.*
//import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController
import com.example.museumemotionapp.utils.FileUtils.logUserLogin
import java.io.File

//import androidx.compose.ui.platform.LocalContext

@Composable
fun ExistingUserScreen(navController: NavController) {
    var selectedUser by remember { mutableStateOf("") }
    var userFolders by remember { mutableStateOf(emptyList<String>()) }
    var dropdownExpanded by remember { mutableStateOf(false) }

    LaunchedEffect(Unit) {
        val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
        val museumEmotionFolder = File(downloadsDir, "MuseumEmotion")

        if (museumEmotionFolder.exists()) {
            userFolders = museumEmotionFolder.list()?.toList() ?: emptyList()
        }
    }

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

        Text("Select an Existing User:")

        Box {
            Button(onClick = { dropdownExpanded = true }) {
                Text(if (selectedUser.isNotEmpty()) selectedUser else "Select an Account")
            }

            DropdownMenu(
                expanded = dropdownExpanded,
                onDismissRequest = { dropdownExpanded = false }
            ) {
                userFolders.forEach { folder ->
                    DropdownMenuItem(
                        text = { Text(folder) },
                        onClick = {
                            selectedUser = folder
                            dropdownExpanded = false
                        }
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        Button(onClick = {
            if (selectedUser.isNotEmpty()) {
                logUserLogin(selectedUser)  // Log user login inside user's own folder
                navController.navigate("artworkSelection/$selectedUser")
            }
        }) {
            Text("Continue / Επόμενο")
        }
    }
}