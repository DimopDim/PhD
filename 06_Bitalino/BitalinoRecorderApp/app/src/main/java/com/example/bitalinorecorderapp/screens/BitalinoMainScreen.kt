package com.example.bitalinorecorderapp.screens

import android.Manifest
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.content.Intent
import android.os.Build
import android.util.Log
import android.widget.Toast
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.example.bitalinorecorderapp.bluetooth.BLEDeviceScanner
import com.example.bitalinorecorderapp.recording.BitalinoRecorder
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch

@Composable
fun BitalinoMainScreen() {
    val context = LocalContext.current
    val bluetoothAdapter: BluetoothAdapter? = BluetoothAdapter.getDefaultAdapter()
    var devices by remember { mutableStateOf(listOf<BluetoothDevice>()) }
    var selectedDevice by remember { mutableStateOf<BluetoothDevice?>(null) }
    var isScanning by remember { mutableStateOf(true) }
    val recorder = remember { BitalinoRecorder(context) }
    var analog by remember { mutableStateOf<List<Int>>(emptyList()) }

    val scope = rememberCoroutineScope()

    // Huawei Auto-Launch Dialog
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

    // 👂 Observe signal values
    LaunchedEffect(Unit) {
        scope.launch {
            recorder.signalValue.collectLatest { values ->
                analog = values ?: emptyList()
            }
        }

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

        devices.forEach { device ->
            Text(
                text = device.name ?: device.address,
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable {
                        selectedDevice = device
                        recorder.startRecording(device)
                    }
                    .padding(8.dp)
            )
        }

        Spacer(modifier = Modifier.height(24.dp))

        if (selectedDevice != null) {
            Button(onClick = {
                recorder.stopRecording()
                selectedDevice = null
            }) {
                Text("Stop Recording")
            }

            Spacer(modifier = Modifier.height(16.dp))

            if (analog.isNotEmpty()) {
                Text("Analog: ${analog.joinToString(", ")}", style = MaterialTheme.typography.bodyLarge)
            }
        }
    }
}
