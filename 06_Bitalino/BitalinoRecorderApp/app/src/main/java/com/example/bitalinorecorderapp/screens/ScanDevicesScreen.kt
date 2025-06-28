package com.example.bitalinorecorderapp.screens

import android.Manifest
import android.bluetooth.BluetoothDevice
import android.content.Intent
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
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.bitalinorecorderapp.bluetooth.BLEDeviceScanner
import com.example.bitalinorecorderapp.bluetooth.BluetoothDeviceScanner
import com.example.bitalinorecorderapp.service.BitalinoRecordingService
import com.example.bitalinorecorderapp.viewmodel.SignalViewModel
import com.example.bitalinorecorderapp.signal.AppSignalBus
import kotlinx.coroutines.flow.collectLatest


@Composable
fun ScanDevicesScreen() {
    val context = LocalContext.current
    val foundDevices = remember { mutableStateListOf<BluetoothDevice>() }
    var isScanning by remember { mutableStateOf(true) }
    var analogValues by remember { mutableStateOf<List<Int>>(emptyList()) }

    val viewModel: SignalViewModel = viewModel()

    // 🔁 Register ViewModel and observe signals
    LaunchedEffect(Unit) {
        AppSignalBus.registerViewModel(viewModel)

        // Collect real-time signal updates
        viewModel.analogSignalFlow.collectLatest { values ->
            analogValues = values
        }

        // Start scanning
        BluetoothDeviceScanner(context) { device ->
            if (!foundDevices.any { it.address == device.address }) {
                foundDevices.add(device)
            }
        }.startScan()

        BLEDeviceScanner(
            context = context,
            onDeviceFound = { device ->
                if (!foundDevices.any { it.address == device.address }) {
                    foundDevices.add(device)
                }
            },
            onScanFinished = { isScanning = false }
        ).startScan()
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        Text("Select a BITalino Device", style = MaterialTheme.typography.titleLarge)

        if (isScanning) {
            Text("Scanning...", modifier = Modifier.padding(8.dp))
        }

        LazyColumn {
            items(foundDevices) { device ->
                val hasPermission = ContextCompat.checkSelfPermission(
                    context,
                    Manifest.permission.BLUETOOTH_CONNECT
                ) == PackageManager.PERMISSION_GRANTED

                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 4.dp)
                        .clickable(enabled = hasPermission) {
                            try {
                                Toast.makeText(context, "Connecting to ${device.name}", Toast.LENGTH_SHORT).show()

                                val intent = Intent(context, BitalinoRecordingService::class.java)
                                intent.putExtra("device_address", device.address)
                                ContextCompat.startForegroundService(context, intent)

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

        Button(
            onClick = {
                val intent = Intent(context, BitalinoRecordingService::class.java)
                context.stopService(intent)
                Toast.makeText(context, "Recording stopped", Toast.LENGTH_SHORT).show()
            },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Stop Recording")
        }

        Spacer(modifier = Modifier.height(24.dp))

        if (analogValues.isNotEmpty()) {
            Text("Analog: ${analogValues.joinToString(", ")}", style = MaterialTheme.typography.bodyLarge)
        }
    }
}
