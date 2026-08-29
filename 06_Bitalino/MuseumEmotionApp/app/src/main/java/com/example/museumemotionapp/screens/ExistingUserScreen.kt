package com.example.museumemotionapp.screens

import android.content.Context
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue   // για το "by remember { ... }"
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavController
import com.example.museumemotionapp.LocalFontScale
import com.example.museumemotionapp.utils.getMuseumRootFolder
import java.io.File

@Composable
fun ExistingUserScreen(
    navController: NavController,
    onUsernameSelected: (String) -> Unit
) {
    val scale = LocalFontScale.current.scale
    val context = LocalContext.current

    // Φορτώνουμε τους users μία φορά όταν ανοίγει η οθόνη
    val users by remember {
        mutableStateOf(getExistingUsers(context))
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Centered content
        Column(
            modifier = Modifier.weight(1f),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Back Button
            Button(onClick = { navController.popBackStack() }) {
                Text("Back / Πίσω", fontSize = 16.sp * scale)
            }

            Spacer(modifier = Modifier.height(16.dp))

            Text("Select an existing user", fontSize = 18.sp * scale)
            Text("Επιλέξτε έναν υπάρχοντα χρήστη", fontSize = 18.sp * scale)

            Spacer(modifier = Modifier.height(16.dp))

            if (users.isEmpty()) {
                Text("No existing users found!", fontSize = 16.sp * scale)
                Text("Δεν βρέθηκαν χρήστες!", fontSize = 16.sp * scale)
            } else {
                LazyColumn(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(300.dp)
                ) {
                    items(users) { username ->
                        Card(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(8.dp)
                                .clickable {
                                    onUsernameSelected(username)
                                    navController.navigate("artworkSelection/$username")
                                }
                        ) {
                            Text(
                                text = username,
                                fontSize = 16.sp * scale,
                                modifier = Modifier.padding(16.dp)
                            )
                        }
                    }
                }
            }
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

/**
 * Επιστρέφει λίστα usernames από τα subfolders του:
 *   /Download/MuseumEmotion
 *
 * Δηλαδή /storage/emulated/0/Download/MuseumEmotion/{username}
 */
fun getExistingUsers(context: Context): List<String> {
    // Ρίζα MuseumEmotion στο Downloads (ορίζεται στο StorageUtils)
    val museumEmotionFolder = getMuseumRootFolder(context)

    return if (museumEmotionFolder.exists()) {
        museumEmotionFolder.listFiles()
            ?.filter { it.isDirectory }
            ?.sortedByDescending { it.lastModified() }
            ?.map { it.name }
            ?: emptyList()
    } else {
        emptyList()
    }
}
