package com.example.bitalinorecorderapp.service

import android.app.*
import android.bluetooth.BluetoothDevice
import android.content.Context
import android.content.Intent
import android.os.*
import android.util.Log
import android.widget.Toast
import androidx.core.app.NotificationCompat
import com.example.bitalinorecorderapp.R
import com.example.bitalinorecorderapp.PersistentActivity // ✅ NEW
import info.plux.pluxapi.bitalino.BITalinoException
import info.plux.pluxapi.bitalino.BITalinoFrame
import info.plux.pluxapi.bitalino.bth.BTHCommunication
import info.plux.pluxapi.bitalino.bth.OnBITalinoDataAvailable
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.*

class BitalinoRecordingService : Service(), OnBITalinoDataAvailable {

    private var bitalinoComm: BTHCommunication? = null
    private var writer: BufferedWriter? = null
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
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val deviceAddress = intent?.getStringExtra("device_address") ?: return START_NOT_STICKY

        // ✅ Launch 1x1 persistent transparent Activity only once
        val persistentIntent = Intent(this, PersistentActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
        }
        startActivity(persistentIntent)

        val downloadsDir = getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS)
        val bitalinoDir = File(downloadsDir, "Bitalino")
        if (!bitalinoDir.exists()) bitalinoDir.mkdirs()

        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val file = File(bitalinoDir, "recording_$timestamp.csv")
        writer = BufferedWriter(FileWriter(file))
        writer?.write("sequence,analog0,analog1,analog2,analog3,analog4,analog5\n")

        bitalinoComm = BTHCommunication(this, this)

        Thread {
            try {
                bitalinoComm?.connect(deviceAddress)
                Thread.sleep(2000)
                bitalinoComm?.start(intArrayOf(0, 1, 2, 3, 4, 5), 100)
                (bitalinoComm as? BTHCommunication)?.sendDigitalOutput(0b0001)
                Log.i(TAG, "Started recording from $deviceAddress")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to start recording", e)
            }
        }.start()

        return START_REDELIVER_INTENT
    }


    override fun onDestroy() {
        try {
            (bitalinoComm as? BTHCommunication)?.sendDigitalOutput(0b0000)
            bitalinoComm?.stop()
            bitalinoComm?.disconnect()
            writer?.flush()
            writer?.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error on stopping service", e)
        }

        releaseWakeLock()
        heartbeatHandler.removeCallbacks(heartbeatRunnable)
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onBITalinoDataAvailable(bitalinoFrame: BITalinoFrame) {
        val sequence = bitalinoFrame.sequence
        val analog = bitalinoFrame.analogArray
        val line = "$sequence,${analog.joinToString(",")}\n"
        writer?.write(line)
    }

    private fun startForegroundServiceWithNotification() {
        val channelId = "bitalino_recording"
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(channelId, "BITalino Recording", NotificationManager.IMPORTANCE_LOW)
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
        wakeLock = powerManager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "BitalinoApp::WakeLockTag")
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
