package com.example.museumemotionapp


import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.navigation.NavType
import androidx.navigation.navArgument
import androidx.compose.ui.unit.sp
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.navigation.compose.*
import com.example.museumemotionapp.screens.*

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

            CompositionLocalProvider(LocalFontScale provides fontSizeLevel) {
                Scaffold(
                    topBar = {
                        TopAppBar(
                            title = { Text("Museum Emotion App", fontSize = (20.sp * fontSizeLevel.scale)) },
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
                        ) { backStackEntry ->
                            val username = backStackEntry.arguments?.getString("username") ?: ""
                            ArtworkListScreen(navController, username)
                        }
                        composable(
                            "artworkDetail/{artworkId}/{username}/{timestampEntry}",
                            arguments = listOf(
                                navArgument("artworkId") { type = NavType.StringType },
                                navArgument("username") { type = NavType.StringType },
                                navArgument("timestampEntry") { type = NavType.LongType }
                            )
                        ) { backStackEntry ->
                            val artworkId = backStackEntry.arguments?.getString("artworkId") ?: ""
                            val username = backStackEntry.arguments?.getString("username") ?: ""
                            val timestampEntry = backStackEntry.arguments?.getLong("timestampEntry") ?: 0L
                            ArtworkDetailScreen(navController, artworkId, username, timestampEntry)
                        }
                        composable("audioPlayback/{artworkId}/{username}") { backStackEntry ->
                            val artworkId = backStackEntry.arguments?.getString("artworkId") ?: ""
                            val username = backStackEntry.arguments?.getString("username") ?: ""
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

                    }
                }
            }
        }
    }
}