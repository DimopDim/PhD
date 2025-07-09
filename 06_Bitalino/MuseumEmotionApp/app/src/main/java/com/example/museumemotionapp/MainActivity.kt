package com.example.museumemotionapp

import android.app.ActivityManager
import android.os.Build
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavType
import androidx.navigation.compose.*
import androidx.navigation.navArgument
import com.example.museumemotionapp.screens.*
import kotlinx.coroutines.launch
import android.os.Environment
import java.io.File


enum class FontSizeLevel(val label: String, val scale: Float) {
    SMALL("A", 0.85f),
    MEDIUM("AA", 1.0f),
    LARGE("AAA", 1.2f)
}

val LocalFontScale = compositionLocalOf { FontSizeLevel.MEDIUM }

@OptIn(ExperimentalMaterial3Api::class)
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        setContent {
            var fontSizeLevel by rememberSaveable { mutableStateOf(FontSizeLevel.MEDIUM) }
            var username by rememberSaveable { mutableStateOf<String?>(null) }
            var showFontSizeMenu by remember { mutableStateOf(false) }
            val navController = rememberNavController()
            val drawerState = rememberDrawerState(DrawerValue.Closed)
            val scope = rememberCoroutineScope()

            CompositionLocalProvider(LocalFontScale provides fontSizeLevel) {
                ModalNavigationDrawer(
                    drawerState = drawerState,
                    gesturesEnabled = true,
                    drawerContent = {
                        ModalDrawerSheet {
                            Text(
                                "Επιλογές",
                                modifier = Modifier.padding(16.dp),
                                style = MaterialTheme.typography.titleMedium
                            )
                            Divider()
                            NavigationDrawerItem(
                                label = { Text("Ξεκλείδωμα Οθόνης") },
                                selected = false,
                                onClick = {
                                    scope.launch {
                                        drawerState.close()
                                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
                                            stopLockTask()
                                        }
                                    }
                                }
                            )
                            NavigationDrawerItem(
                                label = { Text("Κλείδωμα Οθόνης") },
                                selected = false,
                                onClick = {
                                    scope.launch {
                                        drawerState.close()
                                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
                                            Toast.makeText(
                                                this@MainActivity,
                                                "Το app κλειδώθηκε. Για ξεκλείδωμα, επιλέξτε 'Ξεκλείδωμα Οθόνης' από το μενού.",
                                                Toast.LENGTH_LONG
                                            ).show()
                                            startLockTask()
                                        }
                                    }
                                }
                            )

                            NavigationDrawerItem(
                                label = { Text("Επιλογή Χρήστη") },
                                selected = false,
                                onClick = {
                                    scope.launch {
                                        drawerState.close()
                                        navController.navigate("userSelection") {
                                            popUpTo(0) { inclusive = true } // Clears entire backstack
                                        }
                                        username = null
                                    }
                                }
                            )


                            NavigationDrawerItem(
                                label = { Text("Λίστα Έργων Τέχνης") },
                                selected = false,
                                onClick = {
                                    scope.launch {
                                        drawerState.close()
                                        username?.let { user ->
                                            navController.navigate("artworkSelection/$user")
                                        } ?: Toast.makeText(
                                            this@MainActivity,
                                            "Δεν έχει εντοπιστεί χρήστης. Παρακαλώ ξεκινήστε από την αρχική οθόνη.",
                                            Toast.LENGTH_LONG
                                        ).show()
                                    }
                                }
                            )



                            NavigationDrawerItem(
                                label = { Text("Ερωτηματολόγιο 01 : Τεστ προσωπικότητας") },
                                selected = false,
                                onClick = {
                                    scope.launch {
                                        drawerState.close()
                                        username?.let { user ->
                                            val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                                            val bigFiveFile = File(downloadsDir, "MuseumEmotion/$user/bigfive.txt")

                                            if (bigFiveFile.exists()) {
                                                Toast.makeText(
                                                    this@MainActivity,
                                                    "Το ερωτηματολόγιο Big Five έχει ήδη συμπληρωθεί.",
                                                    Toast.LENGTH_LONG
                                                ).show()
                                                // Optional: Show or review answers
                                            } else {
                                                navController.navigate("bigFiveTest/$user")
                                            }
                                        } ?: Toast.makeText(
                                            this@MainActivity,
                                            "Δεν έχει εντοπιστεί χρήστης. Παρακαλώ ξεκινήστε από την αρχική οθόνη.",
                                            Toast.LENGTH_LONG
                                        ).show()
                                    }
                                }
                            )



                            NavigationDrawerItem(
                                label = { Text("Ερωτηματολόγιο 02: Κλίμακα Αλεξιθυμίας") },
                                selected = false,
                                onClick = {
                                    scope.launch {
                                        drawerState.close()
                                        username?.let { user ->
                                            val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                                            val tasFile = File(downloadsDir, "MuseumEmotion/$user/tas.txt")

                                            if (tasFile.exists()) {
                                                Toast.makeText(
                                                    this@MainActivity,
                                                    "Το ερωτηματολόγιο TAS έχει ήδη συμπληρωθεί.",
                                                    Toast.LENGTH_LONG
                                                ).show()
                                            } else {
                                                navController.navigate("tasTest/$user")
                                            }
                                        } ?: Toast.makeText(
                                            this@MainActivity,
                                            "Δεν έχει εντοπιστεί χρήστης. Παρακαλώ ξεκινήστε από την αρχική οθόνη.",
                                            Toast.LENGTH_LONG
                                        ).show()
                                    }
                                }
                            )



                            NavigationDrawerItem(
                                label = { Text("Ερωτηματολόγιο 03: Κλίμακα θετικών και αρνητικών επιδράσεων") },
                                selected = false,
                                onClick = {
                                    scope.launch {
                                        drawerState.close()
                                        username?.let { user ->
                                            val downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                                            val panasFile = File(downloadsDir, "MuseumEmotion/$user/panas.txt")

                                            if (panasFile.exists()) {
                                                Toast.makeText(
                                                    this@MainActivity,
                                                    "Το ερωτηματολόγιο PANAS έχει ήδη συμπληρωθεί.",
                                                    Toast.LENGTH_LONG
                                                ).show()
                                            } else {
                                                navController.navigate("panasTest/$user")
                                            }
                                        } ?: Toast.makeText(
                                            this@MainActivity,
                                            "Δεν έχει εντοπιστεί χρήστης. Παρακαλώ ξεκινήστε από την αρχική οθόνη.",
                                            Toast.LENGTH_LONG
                                        ).show()
                                    }
                                }
                            )







                        }
                    }
                ) {
                    Scaffold(
                        topBar = {
                            TopAppBar(
                                title = {},
                                navigationIcon = {
                                    IconButton(onClick = {
                                        scope.launch { drawerState.open() }
                                    }) {
                                        Icon(Icons.Default.Menu, contentDescription = "Menu")
                                    }
                                },
                                actions = {
                                    Box {
                                        TextButton(onClick = { showFontSizeMenu = true }) {
                                            Text("AAA", fontSize = (14.sp * fontSizeLevel.scale))
                                        }
                                        DropdownMenu(
                                            expanded = showFontSizeMenu,
                                            onDismissRequest = { showFontSizeMenu = false }
                                        ) {
                                            FontSizeLevel.values().forEach { level ->
                                                DropdownMenuItem(
                                                    text = { Text(level.label) },
                                                    onClick = {
                                                        fontSizeLevel = level
                                                        showFontSizeMenu = false
                                                    }
                                                )
                                            }
                                        }
                                    }
                                }
                            )
                        },
                        modifier = Modifier.fillMaxSize()
                    ) { innerPadding ->
                        NavHost(
                            navController = navController,
                            startDestination = "userSelection",
                            modifier = Modifier.padding(innerPadding)
                        ) {
                            composable("userSelection") { UserSelectionScreen(navController) }

                            composable("loginScreen") {
                                LoginScreen(navController) { confirmedUsername ->
                                    username = confirmedUsername
                                    navController.navigate("researchInfo/$confirmedUsername")
                                }
                            }

                            composable("existingUserScreen") {
                                ExistingUserScreen(navController) { selectedUsername ->
                                    username = selectedUsername
                                    navController.navigate("artworkSelection/$selectedUsername")
                                }
                            }

                            composable(
                                "artworkSelection/{username}",
                                arguments = listOf(navArgument("username") { type = NavType.StringType })
                            ) {
                                val uname = it.arguments?.getString("username") ?: ""
                                username = uname
                                ArtworkListScreen(navController, uname)
                            }

                            composable(
                                "artworkDetail/{artworkId}/{username}/{timestampEntry}",
                                arguments = listOf(
                                    navArgument("artworkId") { type = NavType.StringType },
                                    navArgument("username") { type = NavType.StringType },
                                    navArgument("timestampEntry") { type = NavType.LongType }
                                )
                            ) {
                                val artworkId = it.arguments?.getString("artworkId") ?: ""
                                val uname = it.arguments?.getString("username") ?: ""
                                val timestampEntry = it.arguments?.getLong("timestampEntry") ?: 0L
                                username = uname
                                ArtworkDetailScreen(navController, artworkId, uname, timestampEntry)
                            }

                            composable("audioPlayback/{artworkId}/{username}") {
                                val artworkId = it.arguments?.getString("artworkId") ?: ""
                                val uname = it.arguments?.getString("username") ?: ""
                                username = uname
                                AudioPlaybackScreen(navController, artworkId, uname)
                            }

                            composable("researchConsent/{username}",
                                arguments = listOf(navArgument("username") { type = NavType.StringType })
                            ) {
                                val uname = it.arguments?.getString("username") ?: ""
                                username = uname
                                ResearchConsentScreen(navController, uname)
                            }

                            composable("researchInfo/{username}",
                                arguments = listOf(navArgument("username") { type = NavType.StringType })
                            ) {
                                val uname = it.arguments?.getString("username") ?: ""
                                username = uname
                                ResearchInfoScreen(navController, uname)
                            }

                            composable("consentFormScreen/{username}",
                                arguments = listOf(navArgument("username") { type = NavType.StringType })
                            ) {
                                val uname = it.arguments?.getString("username") ?: ""
                                username = uname
                                ConsentFormScreen(navController, uname)
                            }

                            composable("login") {
                                LoginScreen(navController) { confirmedUsername ->
                                    username = confirmedUsername
                                    navController.navigate("researchInfo/$confirmedUsername")
                                }
                            }

                            composable("demographics/{username}") {
                                val uname = it.arguments?.getString("username") ?: ""
                                username = uname
                                DemographicsScreen(navController, uname)
                            }

                            composable(
                                "bigFiveTest/{username}",
                                arguments = listOf(navArgument("username") { type = NavType.StringType })
                            ) {
                                val uname = it.arguments?.getString("username") ?: "unknown"
                                BigFiveScreen(username = uname, navController = navController)
                            }


                            composable(
                                "tasTest/{username}",
                                arguments = listOf(navArgument("username") { type = NavType.StringType })
                            ) {
                                val uname = it.arguments?.getString("username") ?: "unknown"
                                TasScreen(username = uname, navController = navController)
                            }


                            composable(
                                "panasTest/{username}",
                                arguments = listOf(navArgument("username") { type = NavType.StringType })
                            ) {
                                val uname = it.arguments?.getString("username") ?: "unknown"
                                PanasScreen(username = uname, navController = navController)
                            }

                            composable(
                                "panasBegin/{username}",
                                arguments = listOf(navArgument("username") { type = NavType.StringType })
                            ) {
                                val uname = it.arguments?.getString("username") ?: "unknown"
                                PanasScreenBegin(username = uname, navController = navController)
                            }





                        }
                    }
                }
            }
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            startLockTask()
        }
    }

    override fun onBackPressed() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP && isLockTaskModeEnabled()) {
            // Prevent exit while in lock mode
        } else {
            super.onBackPressed()
        }
    }

    private fun isLockTaskModeEnabled(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            val activityManager = getSystemService(ACTIVITY_SERVICE) as ActivityManager
            activityManager.lockTaskModeState != ActivityManager.LOCK_TASK_MODE_NONE
        } else {
            false
        }
    }
}
