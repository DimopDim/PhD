package com.example.museumemotionapp

import android.os.Bundle
import android.os.Environment
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.Alignment
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController
import androidx.navigation.compose.rememberNavController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import java.io.File
import com.example.museumemotionapp.ui.theme.MuseumEmotionAppTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MuseumEmotionAppTheme {
                val navController = rememberNavController()
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    NavHost(
                        navController = navController,
                        startDestination = "userSelection",
                        modifier = Modifier.padding(innerPadding)
                    ) {
                        composable("userSelection") { UserSelectionScreen(navController) }
                        composable("newUser") { LoginScreen(navController) }
                        composable("existingUser") { ExistingUserScreen(navController) }
                        composable("artworkSelection/{username}") { backStackEntry ->
                            ArtworkSelectionScreen(
                                username = backStackEntry.arguments?.getString("username") ?: "",
                                navController = navController
                            )
                        }
                        composable("artworkDetail/{artworkId}") { backStackEntry ->
                            val artworkId = backStackEntry.arguments?.getString("artworkId")
                            val artwork = artworks.find { it.id == artworkId }
                            if (artwork != null) {
                                ArtworkDetailScreen(artwork, navController)
                            }
                        }
                    }
                }
            }
        }
    }
}

// Screen for selecting new or existing user
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

// Screen for new users
@Composable
fun LoginScreen(navController: NavController) {
    var username by remember { mutableStateOf("") }
    var showErrorDialog by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Button(onClick = { navController.popBackStack() }, modifier = Modifier.padding(8.dp)) {
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
                    navController.navigate("artworkSelection/$username")
                }
            }
        }) {
            Text("Continue / Επόμενο")
        }

        if (showErrorDialog) {
            AlertDialog(
                onDismissRequest = { showErrorDialog = false },
                confirmButton = {
                    Button(onClick = { showErrorDialog = false }) {
                        Text("OK")
                    }
                },
                title = { Text("Username Exists") },
                text = { Text("Be careful, this username already exists. Change it before proceeding.") }
            )
        }
    }
}

// Screen for existing users
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
        Button(onClick = { navController.popBackStack() }, modifier = Modifier.padding(8.dp)) {
            Text("Back / Πίσω")
        }

        Spacer(modifier = Modifier.height(16.dp))

        Text("Select an Existing User:")

        Box {
            Button(onClick = { dropdownExpanded = true }, modifier = Modifier.fillMaxWidth()) {
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
                navController.navigate("artworkSelection/$selectedUser")
            }
        }) {
            Text("Continue / Επόμενο")
        }
    }
}

// Screen for selecting an artwork
@Composable
fun ArtworkSelectionScreen(username: String, navController: NavController) {
    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        Text("Choose an Artwork - Επιλέξτε ένα έργο τέχνης ($username)", style = MaterialTheme.typography.headlineMedium)

        LazyColumn {
            items(artworks) { artwork ->
                Button(
                    onClick = { navController.navigate("artworkDetail/${artwork.id}") }
                ) {
                    Text("${artwork.id}. ${artwork.title} - ${artwork.greekTitle}")
                }
                Spacer(modifier = Modifier.height(8.dp))
            }
        }
    }
}

// Screen for artwork details
@Composable
fun ArtworkDetailScreen(artwork: Artwork, navController: NavController) {
    var selectedEmotion by remember { mutableStateOf<String?>(null) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = artwork.title,
            style = MaterialTheme.typography.headlineMedium,
            textAlign = TextAlign.Center
        )

        Spacer(modifier = Modifier.height(16.dp))

        // Emotion Selection
        LazyColumn {
            items(emotions) { emotion ->
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    RadioButton(
                        selected = (selectedEmotion == emotion.id),
                        onClick = { selectedEmotion = emotion.id }
                    )
                    Text(text = "${emotion.id}. ${emotion.greekLabel}")
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        Button(onClick = { navController.popBackStack() }) {
            Text("Back / Πίσω")
        }
    }
}