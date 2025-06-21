package com.example.bitalinorecorderapp.screens

import android.Manifest
import android.bluetooth.BluetoothDevice
import android.content.Context
import android.content.pm.PackageManager
import android.util.Log
import android.widget.Toast
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.example.bitalinorecorderapp.bluetooth.BLEDeviceScanner
import com.example.bitalinorecorderapp.bluetooth.BluetoothDeviceScanner
import com.example.bitalinorecorderapp.recording.BitalinoRecorder
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch

@Composable
fun ScanDevicesScreen() {
    val context = LocalContext.current
    val recorder = remember { BitalinoRecorder(context) }

    val foundDevices = remember { mutableStateListOf<BluetoothDevice>() }
    var isScanning by remember { mutableStateOf(true) }

    val coroutineScope = rememberCoroutineScope()

    var analog by remember { mutableStateOf<List<Int>>(emptyList()) }

    // 🔄 Observe signal updates
    LaunchedEffect(Unit) {
        coroutineScope.launch {
            recorder.signalValue.collectLatest { values ->
                analog = values ?: emptyList()
            }
        }

        BluetoothDeviceScanner(context) { device ->
            if (!foundDevices.any { it.address == device.address }) {
                foundDevices.add(device)
            }
        }.startScan()

        BLEDeviceScanner(
            context,
            onDeviceFound = { device ->
                if (!foundDevices.any { it.address == device.address }) {
                    foundDevices.add(device)
                }
            },
            onScanFinished = { isScanning = false }
        ).startScan()
    }

    Column(modifier = Modifier
        .fillMaxSize()
        .padding(16.dp)) {

        Text("Select a BITalino Device", style = MaterialTheme.typography.titleLarge)

        if (isScanning) {
            Text("Scanning...", modifier = Modifier.padding(8.dp))
        }

        LazyColumn {
            items(foundDevices) { device ->
                val hasPermission = remember {
                    ContextCompat.checkSelfPermission(
                        context,
                        Manifest.permission.BLUETOOTH_CONNECT
                    ) == PackageManager.PERMISSION_GRANTED
                }

                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 4.dp)
                        .clickable(enabled = hasPermission) {
                            try {
                                Toast.makeText(context, "Connecting to ${device.name}", Toast.LENGTH_SHORT).show()
                                recorder.startRecording(device)
                            } catch (e: SecurityException) {
                                Toast.makeText(context, "Missing Bluetooth permission", Toast.LENGTH_LONG).show()
                                Log.e("ScanDevicesScreen", "SecurityException during connect", e)
                            }
                        }
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text("Name: ${if (hasPermission) device.name ?: "Unknown" else "Permission Required"}")
                        Text("Address: ${if (hasPermission) device.address else "Permission Required"}")
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))
        Divider()
        Spacer(modifier = Modifier.height(8.dp))

        // 📊 Real-time signal display
        if (analog.isNotEmpty()) {
            Text("🔢 Analog: ${analog.joinToString(", ")}", style = MaterialTheme.typography.bodyLarge)
        }
    }
}
