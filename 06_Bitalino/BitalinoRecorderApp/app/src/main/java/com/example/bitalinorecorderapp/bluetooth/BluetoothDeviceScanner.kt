package com.example.bitalinorecorderapp.bluetooth

import android.Manifest
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.util.Log
import androidx.core.content.ContextCompat

class BluetoothDeviceScanner(
    private val context: Context,
    private val onDeviceFound: (BluetoothDevice) -> Unit
) {
    private val bluetoothAdapter: BluetoothAdapter? = BluetoothAdapter.getDefaultAdapter()

    fun startScan() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val hasPermission = ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.BLUETOOTH_CONNECT
            ) == PackageManager.PERMISSION_GRANTED

            if (!hasPermission) {
                Log.w("BluetoothScanner", "Missing BLUETOOTH_CONNECT permission")
                return
            }
        }

        bluetoothAdapter?.bondedDevices?.forEach { device ->
            try {
                val name = device.name
                if (name?.startsWith("BITalino") == true) {
                    Log.d("BluetoothScanner", "Found BITalino device: $name")
                    onDeviceFound(device)
                }
            } catch (e: SecurityException) {
                Log.e("BluetoothScanner", "Permission denied when accessing device name", e)
            }
        }
    }

    fun stopScan() {
        // No need to stop bonded device discovery for BTH
    }
}
