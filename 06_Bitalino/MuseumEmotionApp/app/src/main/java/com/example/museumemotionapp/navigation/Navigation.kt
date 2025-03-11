package com.example.museumemotionapp.navigation

import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
//import androidx.navigation.NavHost
import androidx.navigation.NavHostController
import androidx.navigation.compose.composable
import androidx.navigation.compose.NavHost
import com.example.museumemotionapp.screens.*
//import com.example.museumemotionapp.models.Artwork
import com.example.museumemotionapp.models.artworks

@Composable
fun AppNavigation(navController: NavHostController, modifier: Modifier = Modifier) {
    NavHost(
        navController = navController,
        startDestination = "userSelection",
        modifier = modifier
    ) {
        composable("userSelection") { UserSelectionScreen(navController) }
        composable("newUser") { LoginScreen(navController) }
        composable("existingUser") { ExistingUserScreen(navController) }
        composable("artworkSelection/{username}") { backStackEntry ->
            val username = backStackEntry.arguments?.getString("username") ?: ""
            ArtworkSelectionScreen(username, navController)
        }
        composable("artworkDetail/{artworkId}/{username}") { backStackEntry ->
            val artworkId = backStackEntry.arguments?.getString("artworkId")
            val username = backStackEntry.arguments?.getString("username") ?: ""
            val artwork = artworks.find { it.id == artworkId }
            if (artwork != null) {
                ArtworkDetailScreen(artwork, navController, username)
            }
        }
    }
}
