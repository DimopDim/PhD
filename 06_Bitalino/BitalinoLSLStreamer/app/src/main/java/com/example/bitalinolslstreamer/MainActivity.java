package com.example.bitalinolslstreamer;

import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import com.example.bitalinolslstreamer.lsl.LSL;
import com.example.bitalinolslstreamer.bitalino.BITalino;
import com.example.bitalinolslstreamer.bitalino.BITalinoFrame;
import java.util.Arrays;

public class MainActivity extends AppCompatActivity {

    private LSL.StreamOutlet outlet;
    private BITalino bitalino;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        setupLSLStream();
        setupBITalinoConnection();

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
    }

    private void setupLSLStream() {
        // Define the LSL stream
        LSL.StreamInfo info = new LSL.StreamInfo(
                "BITalinoStream",            // Stream name
                "EEG",                       // Content type (modify if needed)
                6,                           // Number of channels (depends on your BITalino)
                LSL.IRREGULAR_RATE,          // Irregular rate or set to your sampling rate
                LSL.ChannelFormat.float32,   // Data type
                "bitalino123"                // Unique ID
        );
        outlet = new LSL.StreamOutlet(info);
    }

    private void setupBITalinoConnection() {
        new Thread(() -> {
            String macAddress = "XX:XX:XX:XX:XX:XX"; // 🔔 Replace with your BITalino MAC address
            int samplingRate = 100; // Hz
            int[] channelsToRead = {0, 1, 2, 3, 4, 5}; // BITalino channels

            bitalino = new BITalino(macAddress);
            try {
                bitalino.connect();
                bitalino.start(samplingRate, channelsToRead);

                while (true) {
                    BITalinoFrame[] frames = bitalino.read(10); // Read 10 frames at a time
                    for (BITalinoFrame frame : frames) {
                        float[] sample = new float[6];
                        for (int i = 0; i < 6; i++) {
                            sample[i] = frame.analog[i];
                        }
                        outlet.push_sample(sample); // Send to LabRecorder
                    }
                }

            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                try {
                    bitalino.stop();
                    bitalino.disconnect();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }).start();
    }
}
