package com.example.bitalinorecorderapp

import android.Manifest
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.content.Intent
import android.net.Uri
import android.os.Environment
import android.os.PowerManager
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.core.content.ContextCompat
import com.example.bitalinorecorderapp.screens.BitalinoMainScreen

class MainActivity : ComponentActivity() {

    private val requestPermissionsLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
            // Handle permission results if needed
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Runtime permissions
        val permissions = mutableListOf<String>()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            permissions.add(Manifest.permission.BLUETOOTH_SCAN)
            permissions.add(Manifest.permission.BLUETOOTH_CONNECT)
        } else {
            permissions.add(Manifest.permission.BLUETOOTH)
            permissions.add(Manifest.permission.BLUETOOTH_ADMIN)
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            permissions.add(Manifest.permission.ACCESS_FINE_LOCATION)
        } else {
            permissions.add(Manifest.permission.ACCESS_COARSE_LOCATION)
        }

        if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
            permissions.add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
        }

        // Manage all files (Downloads access)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            if (!Environment.isExternalStorageManager()) {
                Toast.makeText(this, "Requesting access to Downloads folder...", Toast.LENGTH_SHORT).show()
                val intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION)
                intent.data = Uri.parse("package:$packageName")
                startActivity(intent)
            }
        }

        // Request missing permissions
        val missingPermissions = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != android.content.pm.PackageManager.PERMISSION_GRANTED
        }

        if (missingPermissions.isNotEmpty()) {
            requestPermissionsLauncher.launch(missingPermissions.toTypedArray())
        }

        // Ask to ignore battery optimizations
        requestIgnoreBatteryOptimizations()

        // Launch persistent 1x1 invisible activity to keep app alive
        launchPersistentActivity()

        setContent {
            BitalinoMainScreen()
        }
    }

    private fun requestIgnoreBatteryOptimizations() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            val powerManager = getSystemService(POWER_SERVICE) as PowerManager
            if (!powerManager.isIgnoringBatteryOptimizations(packageName)) {
                val intent = Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS)
                intent.data = Uri.parse("package:$packageName")
                startActivity(intent)

                Toast.makeText(
                    this,
                    "Please allow ignoring battery optimizations for uninterrupted background recording.",
                    Toast.LENGTH_LONG
                ).show()

                // Show dialog for Huawei users
                showHuaweiAutoLaunchDialog()
            }
        }
    }

    private fun showHuaweiAutoLaunchDialog() {
        AlertDialog.Builder(this)
            .setTitle("Allow Auto-launch on Huawei")
            .setMessage("To ensure the app records signals in the background without interruptions, please allow it to auto-launch and run in the background.")
            .setPositiveButton("Open Settings") { _, _ -> openHuaweiAutoLaunchSettings() }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun openHuaweiAutoLaunchSettings() {
        try {
            val intent = Intent()
            intent.setClassName(
                "com.huawei.systemmanager",
                "com.huawei.systemmanager.startupmgr.ui.StartupNormalAppListActivity"
            )
            startActivity(intent)
        } catch (e: Exception) {
            Toast.makeText(
                this,
                "Huawei-specific settings not found. Please check battery and app launch settings manually.",
                Toast.LENGTH_LONG
            ).show()
        }
    }

    private fun launchPersistentActivity() {
        val intent = Intent(this, PersistentActivity::class.java)
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP)
        startActivity(intent)
    }
}
