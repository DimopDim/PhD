package com.example.bitalinorecorderapp.service

import android.app.*
import android.content.Context
import android.content.Intent
import android.os.*
import android.util.Log
import android.widget.Toast
import androidx.core.app.NotificationCompat
import com.example.bitalinorecorderapp.R
import com.example.bitalinorecorderapp.PersistentActivity
import com.example.bitalinorecorderapp.signal.AppSignalBus
import info.plux.pluxapi.bitalino.BITalinoFrame
import info.plux.pluxapi.bitalino.bth.BTHCommunication
import info.plux.pluxapi.bitalino.bth.OnBITalinoDataAvailable
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.*

class BitalinoRecordingService : Service(), OnBITalinoDataAvailable {

    companion object {
        const val ACTION_CONNECT_ONLY = "CONNECT_ONLY"
        const val ACTION_START_RECORDING = "START_RECORDING"
        const val ACTION_CONNECTED = "com.example.bitalinorecorderapp.BITALINO_CONNECTED"
    }

    private var bitalinoComm: BTHCommunication? = null
    private var writer: BufferedWriter? = null
    private var isDeviceConnected = false
    private var sampleRate: Int = 100
    private var deviceAddress: String? = null
    private val TAG = "BitalinoService"
    private var wakeLock: PowerManager.WakeLock? = null

    private val heartbeatHandler = Handler(Looper.getMainLooper())
    private val heartbeatRunnable = object : Runnable {
        override fun run() {
            Log.d(TAG, "Heartbeat – keeping service alive.")
            heartbeatHandler.postDelayed(this, 5 * 60 * 1000)
        }
    }

    override fun onCreate() {
        super.onCreate()
        acquireWakeLock()
        startForegroundServiceWithNotification()
        heartbeatHandler.post(heartbeatRunnable)

        val persistentIntent = Intent(this, PersistentActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
        }
        startActivity(persistentIntent)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val action = intent?.action
        deviceAddress = intent?.getStringExtra("device_address")
        sampleRate = intent?.getIntExtra("sampling_rate", 100) ?: 100

        if (deviceAddress == null) {
            Log.e(TAG, "No device address provided.")
            stopSelf()
            return START_NOT_STICKY
        }

        when (action) {
            ACTION_CONNECT_ONLY -> connectToDevice()
            ACTION_START_RECORDING -> startRecording(intent)
        }

        return START_REDELIVER_INTENT
    }

    private fun connectToDevice() {
        Thread {
            try {
                bitalinoComm = BTHCommunication(this, this)
                bitalinoComm?.connect(deviceAddress)
                isDeviceConnected = true

                sendBroadcast(Intent(ACTION_CONNECTED))
                Log.i(TAG, "Connected to BITalino: $deviceAddress")

                Handler(Looper.getMainLooper()).post {
                    Toast.makeText(this, "Connected to BITalino", Toast.LENGTH_SHORT).show()
                }

            } catch (e: Exception) {
                Log.e(TAG, "Connection failed", e)
                Handler(Looper.getMainLooper()).post {
                    Toast.makeText(this, "Connection failed", Toast.LENGTH_LONG).show()
                }
            }
        }.start()
    }

    private fun startRecording(intent: Intent?) {
        if (!isDeviceConnected || deviceAddress == null) {
            Log.e(TAG, "Device not connected. Cannot start recording.")
            return
        }

        val sensorLabels = (1..6).map { i ->
            intent?.getStringExtra("sensor_A$i") ?: ""
        }

        val downloadsDir = getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS)
        val bitalinoDir = File(downloadsDir, "Bitalino")
        if (!bitalinoDir.exists()) bitalinoDir.mkdirs()

        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val file = File(bitalinoDir, "recording_${sampleRate}Hz_$timestamp.txt")
        writer = BufferedWriter(FileWriter(file))

        val macAddress = deviceAddress!!.uppercase()
        val headerJson = """
# OpenSignals Text File Format
# {"$macAddress":{
  "sensor":["ACC","EDA","ECG"],
  "device name":"$macAddress",
  "column":["nSeq","A1","A2","A3","A4","A5","A6"],
  "sync interval":2,
  "time":"${SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date())}",
  "comments":"",
  "device connection":"BTH$macAddress",
  "channels":[1,2,3,4,5,6],
  "date":"${SimpleDateFormat("yyyy-MM-dd", Locale.getDefault()).format(Date())}",
  "mode":0,
  "firmware version":0,
  "device":"bitalino_rev",
  "position":0,
  "sampling rate":$sampleRate,
  "label":${sensorLabels.map { "\"$it\"" }},
  "resolution":[10,10,10,10,10,10],
  "special":[{},{},{},{},{},{}]
}}
# EndOfHeader
""".trimIndent()

        writer?.write(headerJson + "\n")

        Thread {
            try {
                bitalinoComm?.start(intArrayOf(0, 1, 2, 3, 4, 5), sampleRate)
                (bitalinoComm as? BTHCommunication)?.sendDigitalOutput(0b1111)

                Handler(Looper.getMainLooper()).post {
                    Toast.makeText(this, "Recording started at $sampleRate Hz", Toast.LENGTH_SHORT).show()
                }

            } catch (e: Exception) {
                Log.e(TAG, "Failed to start recording", e)
                Handler(Looper.getMainLooper()).post {
                    Toast.makeText(this, "Failed to start recording", Toast.LENGTH_LONG).show()
                }
            }
        }.start()
    }

    override fun onDestroy() {
        try {
            (bitalinoComm as? BTHCommunication)?.sendDigitalOutput(0b0000)
            bitalinoComm?.stop()
            bitalinoComm?.disconnect()

            Thread.sleep(500)
            writer?.flush()
            writer?.close()

            Handler(Looper.getMainLooper()).post {
                Toast.makeText(this, "Recording stopped", Toast.LENGTH_SHORT).show()
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error on stopping service", e)
        }

        releaseWakeLock()
        heartbeatHandler.removeCallbacks(heartbeatRunnable)
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onBITalinoDataAvailable(bitalinoFrame: BITalinoFrame) {
        try {
            val sequence = bitalinoFrame.sequence
            val analog = bitalinoFrame.analogArray

            // Keep analog values as-is (no inversion)
            val processedAnalog = analog.copyOf()

            // Write to file
            val line = buildString {
                append(sequence)
                processedAnalog.forEach { value ->
                    append("\t")
                    append(value)
                }
                append("\n")
            }

            writer?.write(line)
            writer?.flush()

            // Emit original values
            CoroutineScope(Dispatchers.Main).launch {
                AppSignalBus.emit(processedAnalog.toList())
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error writing or emitting data frame", e)
        }
    }



    private fun startForegroundServiceWithNotification() {
        val channelId = "bitalino_recording"
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                channelId,
                "BITalino Recording",
                NotificationManager.IMPORTANCE_LOW
            )
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(channel)
        }

        val notification: Notification = NotificationCompat.Builder(this, channelId)
            .setContentTitle("BITalino Recording")
            .setContentText("Recording biosignals...")
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setOngoing(true)
            .build()

        startForeground(1, notification)
    }

    private fun acquireWakeLock() {
        val powerManager = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = powerManager.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "BitalinoApp::WakeLockTag"
        )
        wakeLock?.acquire()
        Log.i(TAG, "WakeLock acquired")
    }

    private fun releaseWakeLock() {
        wakeLock?.let {
            if (it.isHeld) {
                it.release()
                Log.i(TAG, "WakeLock released")
            }
        }
    }
}