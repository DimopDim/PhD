package com.example.bitalinorecorderapp.recording

import android.bluetooth.BluetoothDevice
import android.content.Context
import android.os.Environment
import android.util.Log
import info.plux.pluxapi.bitalino.BITalinoFrame
import info.plux.pluxapi.bitalino.BITalinoException
import info.plux.pluxapi.bitalino.bth.BTHCommunication
import info.plux.pluxapi.bitalino.bth.OnBITalinoDataAvailable
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.*

class BitalinoRecorder(private val context: Context) : OnBITalinoDataAvailable {

    private var bitalinoComm: BTHCommunication? = null
    private var writer: BufferedWriter? = null
    private val TAG = "BitalinoRecorder"

    fun startRecording(device: BluetoothDevice) {
        try {
            // Create output file
            val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
            val dir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "Bitalino")
            if (!dir.exists()) dir.mkdirs()

            val file = File(dir, "bitalino_recording_$timeStamp.csv")
            writer = BufferedWriter(FileWriter(file))

            // Initialize BTHCommunication
            bitalinoComm = BTHCommunication(context, this)

            // 1. Connect to the MAC address
            bitalinoComm?.connect(device.address)

            // 2. Start acquisition (after connection)
            val analogChannels = intArrayOf(0) // Adjust depending on your setup
            val sampleRate = 100 // Supported values: 1, 10, 100, 1000, etc.
            bitalinoComm?.start(analogChannels, sampleRate)

            Log.d(TAG, "Started recording to ${file.absolutePath}")
        } catch (e: BITalinoException) {
            Log.e(TAG, "BITalino error: ${e.message}", e)
        } catch (e: Exception) {
            Log.e(TAG, "Unexpected error during recording start", e)
        }
    }

    fun stopRecording() {
        try {
            bitalinoComm?.stop()
            bitalinoComm?.disconnect()
            writer?.flush()
            writer?.close()
            Log.d(TAG, "Stopped recording and closed file")
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping BITalino", e)
        }
    }

    override fun onBITalinoDataAvailable(frame: BITalinoFrame?) {
        frame?.let {
            try {
                val timestamp = it.getSequence()
                val analogValues = it.getAnalogArray().joinToString(",")
                val line = "$timestamp,$analogValues\n"
                writer?.write(line)
            } catch (e: Exception) {
                Log.e(TAG, "Error writing frame", e)
            }
        }
    }
}
