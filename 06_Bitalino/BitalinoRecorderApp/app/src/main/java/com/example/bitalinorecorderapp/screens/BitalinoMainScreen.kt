package com.example.bitalinorecorderapp.screens

import android.Manifest
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Build
import android.util.Log
import android.widget.Toast
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.example.bitalinorecorderapp.bluetooth.BLEDeviceScanner
import com.example.bitalinorecorderapp.service.BitalinoRecordingService
import kotlinx.coroutines.launch

const val ACTION_BITALINO_CONNECTED = "com.example.bitalinorecorderapp.BITALINO_CONNECTED"

@Composable
fun BitalinoMainScreen() {
    val context = LocalContext.current
    val bluetoothAdapter: BluetoothAdapter? = BluetoothAdapter.getDefaultAdapter()
    var devices by remember { mutableStateOf(listOf<BluetoothDevice>()) }
    var selectedDevice by remember { mutableStateOf<BluetoothDevice?>(null) }
    var isScanning by remember { mutableStateOf(true) }
    var isConnected by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()

    val samplingRates = listOf(1000, 100, 10)
    var selectedRate by remember { mutableStateOf(100) }
    var showRateMenu by remember { mutableStateOf(false) }

    var showHuaweiDialog by remember { mutableStateOf(true) }

    if (showHuaweiDialog) {
        AlertDialog(
            onDismissRequest = { showHuaweiDialog = false },
            title = { Text("Huawei Auto-Launch") },
            text = {
                Text("To ensure background recording works reliably, please enable Auto-launch, Secondary launch, and Background activity for this app. Do you want to open these settings now?")
            },
            confirmButton = {
                TextButton(onClick = {
                    showHuaweiDialog = false
                    try {
                        val intent = Intent().apply {
                            setClassName(
                                "com.huawei.systemmanager",
                                "com.huawei.systemmanager.startupmgr.ui.StartupNormalAppListActivity"
                            )
                        }
                        context.startActivity(intent)
                    } catch (e: Exception) {
                        Toast.makeText(
                            context,
                            "Huawei-specific settings not available. Please check manually.",
                            Toast.LENGTH_LONG
                        ).show()
                    }
                }) {
                    Text("Yes")
                }
            },
            dismissButton = {
                TextButton(onClick = { showHuaweiDialog = false }) {
                    Text("No")
                }
            }
        )
    }

    @Suppress("UnspecifiedRegisterReceiverFlag", "MissingPermission")
    DisposableEffect(Unit) {
        val receiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context?, intent: Intent?) {
                isConnected = true
                Toast.makeText(context, "BITalino connected!", Toast.LENGTH_SHORT).show()
            }
        }

        val filter = IntentFilter(ACTION_BITALINO_CONNECTED)
        context.registerReceiver(receiver, filter)

        onDispose {
            context.unregisterReceiver(receiver)
        }
    }

    LaunchedEffect(Unit) {
        val hasPermission = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            ContextCompat.checkSelfPermission(context, Manifest.permission.BLUETOOTH_CONNECT) ==
                    android.content.pm.PackageManager.PERMISSION_GRANTED &&
                    ContextCompat.checkSelfPermission(context, Manifest.permission.BLUETOOTH_SCAN) ==
                    android.content.pm.PackageManager.PERMISSION_GRANTED
        } else {
            ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION) ==
                    android.content.pm.PackageManager.PERMISSION_GRANTED
        }

        if (hasPermission) {
            BLEDeviceScanner(
                context = context,
                onDeviceFound = { device ->
                    if (!devices.any { it.address == device.address }) {
                        devices = devices + device
                    }
                },
                onScanFinished = { isScanning = false }
            ).startScan()
        } else {
            Log.w("BitalinoMainScreen", "Missing required Bluetooth or Location permissions.")
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        Text("Select BITalino Device", style = MaterialTheme.typography.titleLarge)
        Spacer(modifier = Modifier.height(16.dp))

        if (isScanning) {
            Text("Scanning...", style = MaterialTheme.typography.bodyLarge)
            Spacer(modifier = Modifier.height(16.dp))
        }

        Text("Sampling Rate (Hz):", style = MaterialTheme.typography.bodyMedium)
        Box {
            Button(onClick = { showRateMenu = true }) {
                Text("$selectedRate Hz")
            }
            DropdownMenu(
                expanded = showRateMenu,
                onDismissRequest = { showRateMenu = false }
            ) {
                samplingRates.forEach { rate ->
                    DropdownMenuItem(
                        text = { Text("$rate Hz") },
                        onClick = {
                            selectedRate = rate
                            showRateMenu = false
                        }
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(24.dp))

        devices.forEach { device ->
            Text(
                text = device.name ?: device.address,
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable {
                        selectedDevice = device
                        isConnected = false

                        val intent = Intent(context, BitalinoRecordingService::class.java).apply {
                            action = BitalinoRecordingService.ACTION_CONNECT_ONLY
                            putExtra("device_address", device.address)
                            putExtra("sampling_rate", selectedRate)
                        }
                        ContextCompat.startForegroundService(context, intent)

                        Toast.makeText(
                            context,
                            "Connecting to ${device.address} at $selectedRate Hz...",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                    .padding(8.dp)
            )
        }

        Spacer(modifier = Modifier.height(24.dp))

        if (isConnected && selectedDevice != null) {
            Button(onClick = {
                val intent = Intent(context, BitalinoRecordingService::class.java).apply {
                    action = BitalinoRecordingService.ACTION_START_RECORDING
                    putExtra("device_address", selectedDevice!!.address)
                    putExtra("sampling_rate", selectedRate)
                }
                ContextCompat.startForegroundService(context, intent)
                Toast.makeText(context, "Recording started!", Toast.LENGTH_SHORT).show()
            }) {
                Text("Start Recording")
            }
        }

        if (selectedDevice != null) {
            Button(
                onClick = {
                    val intent = Intent(context, BitalinoRecordingService::class.java)
                    context.stopService(intent)
                    Toast.makeText(context, "Recording stopped", Toast.LENGTH_SHORT).show()
                    selectedDevice = null
                    isConnected = false
                },
                modifier = Modifier.padding(top = 16.dp)
            ) {
                Text("Stop Recording")
            }
        }

        Spacer(modifier = Modifier.weight(1f)) // Push footer to the bottom

        Divider(modifier = Modifier.padding(vertical = 8.dp))

        Text(
            text = "© 2025 MMAI Team | University of the Aegean",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 8.dp)
        )
    }
}