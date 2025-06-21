package com.example.bitalinorecorderapp.bluetooth

import android.Manifest
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.le.*
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.util.Log
import androidx.core.content.ContextCompat

class BLEDeviceScanner(
    private val context: Context,
    private val onDeviceFound: (BluetoothDevice) -> Unit,
    private val onScanFinished: () -> Unit
) {
    private val bluetoothAdapter: BluetoothAdapter? = BluetoothAdapter.getDefaultAdapter()
    private val scanner: BluetoothLeScanner? = bluetoothAdapter?.bluetoothLeScanner
    private var hasConnected = false

    private val scanCallback = object : ScanCallback() {
        override fun onScanResult(callbackType: Int, result: ScanResult) {
            try {
                val device = result.device
                if (!hasConnected && device?.name?.contains("BITalino", ignoreCase = true) == true) {
                    Log.d("BLEDeviceScanner", "✅ Found device: ${device.name} - ${device.address}")
                    hasConnected = true
                    stopScan()
                    onDeviceFound(device)
                }
            } catch (e: SecurityException) {
                Log.e("BLEDeviceScanner", "❌ SecurityException when accessing device info", e)
            }
        }

        override fun onScanFailed(errorCode: Int) {
            Log.e("BLEDeviceScanner", "❌ Scan failed with error: $errorCode")
            onScanFinished()
        }
    }

    fun startScan(scanDurationMillis: Long = 10000L) {
        hasConnected = false

        if (!hasPermissions()) {
            Log.e("BLEDeviceScanner", "❌ Missing Bluetooth or Location permissions")
            onScanFinished()
            return
        }

        if (scanner == null) {
            Log.e("BLEDeviceScanner", "❌ BLE Scanner not available")
            onScanFinished()
            return
        }

        try {
            Log.d("BLEDeviceScanner", "🔍 Starting BLE scan...")
            scanner.startScan(scanCallback)

            Handler(Looper.getMainLooper()).postDelayed({
                if (!hasConnected) stopScan()
            }, scanDurationMillis)
        } catch (e: SecurityException) {
            Log.e("BLEDeviceScanner", "❌ SecurityException during startScan", e)
            onScanFinished()
        }
    }

    fun stopScan() {
        try {
            Log.d("BLEDeviceScanner", "🛑 Stopping BLE scan")
            scanner?.stopScan(scanCallback)
        } catch (e: SecurityException) {
            Log.e("BLEDeviceScanner", "⚠️ SecurityException during stopScan", e)
        } finally {
            onScanFinished()
        }
    }

    private fun hasPermissions(): Boolean {
        val permissionsToCheck = mutableListOf(
            Manifest.permission.ACCESS_FINE_LOCATION
        )

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            permissionsToCheck.add(Manifest.permission.BLUETOOTH_SCAN)
        } else {
            permissionsToCheck.add(Manifest.permission.BLUETOOTH)
            permissionsToCheck.add(Manifest.permission.BLUETOOTH_ADMIN)
        }

        return permissionsToCheck.all {
            ContextCompat.checkSelfPermission(context, it) == PackageManager.PERMISSION_GRANTED
        }
    }
}
