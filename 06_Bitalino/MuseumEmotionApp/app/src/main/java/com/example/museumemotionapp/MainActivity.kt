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
            var showFontSizeMenu by remember { mutableStateOf(false) }
            val navController = rememberNavController()
            val drawerState = rememberDrawerState(DrawerValue.Closed)
            val scope = rememberCoroutineScope()

            CompositionLocalProvider(LocalFontScale provides fontSizeLevel) {
                ModalNavigationDrawer(
                    drawerState = drawerState,
                    drawerContent = {
                        ModalDrawerSheet {
                            Text(
                                "Επιλογές",
                                modifier = Modifier.padding(16.dp),
                                style = MaterialTheme.typography.titleMedium
                            )
                            Divider()
                            NavigationDrawerItem(
                                label = { Text("Unpin") },
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
                                label = { Text("Pin") },
                                selected = false,
                                onClick = {
                                    scope.launch {
                                        drawerState.close()
                                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
                                            Toast.makeText(
                                                this@MainActivity,
                                                "Το app κλειδώθηκε. Για ξεκλείδωμα, επιλέξτε 'Unpin' από το μενού.",
                                                Toast.LENGTH_LONG
                                            ).show()
                                            startLockTask()
                                        }
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
                            composable("loginScreen") { LoginScreen(navController) }
                            composable("existingUserScreen") { ExistingUserScreen(navController) }
                            composable(
                                "artworkSelection/{username}",
                                arguments = listOf(navArgument("username") { type = NavType.StringType })
                            ) {
                                val username = it.arguments?.getString("username") ?: ""
                                ArtworkListScreen(navController, username)
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
                                val username = it.arguments?.getString("username") ?: ""
                                val timestampEntry = it.arguments?.getLong("timestampEntry") ?: 0L
                                ArtworkDetailScreen(navController, artworkId, username, timestampEntry)
                            }
                            composable("audioPlayback/{artworkId}/{username}") {
                                val artworkId = it.arguments?.getString("artworkId") ?: ""
                                val username = it.arguments?.getString("username") ?: ""
                                AudioPlaybackScreen(navController, artworkId, username)
                            }
                            composable("researchConsent/{username}",
                                arguments = listOf(navArgument("username") { type = NavType.StringType })
                            ) {
                                val username = it.arguments?.getString("username") ?: ""
                                ResearchConsentScreen(navController, username)
                            }
                            composable("researchInfo/{username}",
                                arguments = listOf(navArgument("username") { type = NavType.StringType })
                            ) {
                                val username = it.arguments?.getString("username") ?: ""
                                ResearchInfoScreen(navController, username)
                            }
                            composable("consentFormScreen/{username}",
                                arguments = listOf(navArgument("username") { type = NavType.StringType })
                            ) {
                                val username = it.arguments?.getString("username") ?: ""
                                ConsentFormScreen(navController, username)
                            }
                            composable("login") {
                                LoginScreen(navController)
                            }

                            composable("demographics/{username}") { backStackEntry ->
                                val username = backStackEntry.arguments?.getString("username") ?: ""
                                DemographicsScreen(navController, username)
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
            // Do nothing to prevent exit
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
