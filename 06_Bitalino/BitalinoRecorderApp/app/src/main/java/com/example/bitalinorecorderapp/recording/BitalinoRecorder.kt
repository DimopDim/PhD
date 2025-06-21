package com.example.bitalinorecorderapp.recording

import android.bluetooth.BluetoothDevice
import android.content.Context
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.widget.Toast
import info.plux.pluxapi.bitalino.BITalinoException
import info.plux.pluxapi.bitalino.BITalinoFrame
import info.plux.pluxapi.bitalino.bth.BTHCommunication
import info.plux.pluxapi.bitalino.bth.OnBITalinoDataAvailable
import kotlinx.coroutines.flow.MutableStateFlow
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.*

class BitalinoRecorder(private val context: Context) : OnBITalinoDataAvailable {

    private var bitalinoComm: BTHCommunication? = null
    private var writer: BufferedWriter? = null
    private val TAG = "BitalinoRecorder"

    val signalValue = MutableStateFlow<List<Int>?>(null)

    fun startRecording(device: BluetoothDevice) {
        try {
            val downloadsDir = context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS)
            val bitalinoDir = File(downloadsDir, "Bitalino")
            if (!bitalinoDir.exists() && !bitalinoDir.mkdirs()) {
                Log.e(TAG, "❌ Failed to create directory: ${bitalinoDir.absolutePath}")
                Handler(Looper.getMainLooper()).post {
                    Toast.makeText(context, "Failed to create folder", Toast.LENGTH_SHORT).show()
                }
                return
            }

            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
            val file = File(bitalinoDir, "bitalino_recording_$timestamp.csv")
            writer = BufferedWriter(FileWriter(file))
            writer?.write("sequence,analog0,analog1,analog2,analog3,analog4,analog5\n")

            bitalinoComm = BTHCommunication(context, this)
            bitalinoComm?.connect(device.address)

            Log.d(TAG, "🔌 Connecting to ${device.address}...")

            Thread {
                var started = false
                for (attempt in 1..10) {
                    try {
                        val sampleRate = 100
                        val analogChannels = intArrayOf(0, 1, 2, 3, 4, 5)
                        bitalinoComm?.start(analogChannels, sampleRate)

                        // ✅ Turn on LED at O1 (digital output bitmask 0b0001 = 1)
                        (bitalinoComm as? BTHCommunication)?.sendDigitalOutput(0b1111) // Turn ON O1

                        started = true
                        Log.d(TAG, "▶️ BITalino started recording (attempt $attempt)")
                        break
                    } catch (e: Exception) {
                        Log.w(TAG, "⏱ Waiting for BITalino connection (attempt $attempt)...")
                        Thread.sleep(1000)
                    }
                }

                Handler(Looper.getMainLooper()).post {
                    if (started) {
                        Toast.makeText(context, "Recording started", Toast.LENGTH_SHORT).show()
                    } else {
                        Log.e(TAG, "❌ Failed to start BITalino after retries")
                        Toast.makeText(context, "Failed to start recording", Toast.LENGTH_SHORT).show()
                    }
                }
            }.start()

        } catch (e: BITalinoException) {
            Log.e(TAG, "❌ BITalino error: ${e.message}", e)
        } catch (e: Exception) {
            Log.e(TAG, "❌ Unexpected error during recording start", e)
        }
    }

    fun stopRecording() {
        try {
            // ✅ Turn off LED at O1 (bitmask 0)
            (bitalinoComm as? BTHCommunication)?.sendDigitalOutput(0b0001)  // Turn OFF all LEDs


            bitalinoComm?.stop()
            bitalinoComm?.disconnect()
            writer?.flush()
            writer?.close()

            Log.d(TAG, "✅ Recording stopped and file closed")
            Handler(Looper.getMainLooper()).post {
                Toast.makeText(context, "Recording stopped", Toast.LENGTH_SHORT).show()
            }
        } catch (e: Exception) {
            Log.e(TAG, "⚠️ Error stopping BITalino", e)
        }
    }

    override fun onBITalinoDataAvailable(bitalinoFrame: BITalinoFrame) {
        try {
            val sequence = bitalinoFrame.sequence
            val analogValues = bitalinoFrame.analogArray
            signalValue.value = analogValues.toList()

            val line = "$sequence,${analogValues.joinToString(",")}\n"
            writer?.write(line)
        } catch (e: Exception) {
            Log.e(TAG, "⚠️ Error writing frame", e)
        }
    }
}
