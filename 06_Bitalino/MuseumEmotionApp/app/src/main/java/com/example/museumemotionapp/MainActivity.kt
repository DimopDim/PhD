package com.example.museumemotionapp

import android.os.Bundle
import androidx.compose.ui.Modifier
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.*
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.example.museumemotionapp.screens.*

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            val navController = rememberNavController()
            Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
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
                }
            }
        }
    }
}
