// --- MainActivity.java ---
package com.ajay.smishing;

import android.Manifest;
import android.content.IntentFilter;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.provider.Telephony;
import android.util.Log;
import android.widget.ArrayAdapter;
import android.widget.ListView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "PhishingDetector";
    private static final int SMS_PERMISSION_CODE = 100;

    private ListView smsListView;
    private ArrayAdapter<String> smsAdapter;
    private ArrayList<String> smsMessages;
    private SmsReceiver smsReceiver;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        smsListView = findViewById(R.id.smsListView);
        smsMessages = new ArrayList<>();
        smsAdapter = new ArrayAdapter<>(this, android.R.layout.simple_list_item_1, smsMessages);
        smsListView.setAdapter(smsAdapter);

        smsReceiver = new SmsReceiver();

        // Request SMS permissions at runtime
        if (checkSmsPermissions()) {
            registerSmsReceiver();
        } else {
            requestSmsPermissions();
        }
    }

    // Check if SMS permissions are granted
    private boolean checkSmsPermissions() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.RECEIVE_SMS) == PackageManager.PERMISSION_GRANTED && ContextCompat.checkSelfPermission(this, Manifest.permission.READ_SMS) == PackageManager.PERMISSION_GRANTED;
    }

    // Request SMS permissions
    private void requestSmsPermissions() {
        ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.RECEIVE_SMS, Manifest.permission.READ_SMS}, SMS_PERMISSION_CODE);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == SMS_PERMISSION_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED && grantResults[1] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "SMS permissions granted!", Toast.LENGTH_SHORT).show();
                registerSmsReceiver();
            } else {
                Toast.makeText(this, "SMS permissions denied. App may not function correctly.", Toast.LENGTH_LONG).show();
            }
        }
    }

    // Register the SMS BroadcastReceiver
    private void registerSmsReceiver() {
        IntentFilter filter = new IntentFilter(Telephony.Sms.Intents.SMS_RECEIVED_ACTION);
        registerReceiver(smsReceiver, filter);
        Log.d(TAG, "SMS Receiver registered.");
    }

    // Unregister the SMS BroadcastReceiver when the activity is destroyed
    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (smsReceiver != null) {
            unregisterReceiver(smsReceiver);
            Log.d(TAG, "SMS Receiver unregistered.");
        }
    }

    // Method to update the UI with new SMS messages and detection results
    public void updateSmsDisplay(String sender, String message) {
        // --- PLACEHOLDER FOR ML MODEL INFERENCE AND XAI ---
        // In this section, you will:
        // 1. Preprocess the 'message' string.
        // 2. Pass the preprocessed message to your loaded TFLite model for prediction.
        // 3. Get the prediction result (e.g., "phishing" or "ham").
        // 4. If "phishing", apply your XAI logic to the 'message' to get importance scores.
        // 5. Format the message with XAI highlights (e.g., using SpannableString for colored text).

        String detectionStatus = " (Status: Processing...)"; // Default placeholder
        // Example of how you might update based on a hypothetical model result:
        // if (model.predict(message) == PHISHING) {
        //     detectionStatus = " (Status: PHISHING DETECTED!)";
        //     // Apply XAI here and get highlighted text
        //     // String highlightedMessage = xai.highlight(message);
        //     // smsMessages.add("From: " + sender + "\n" + highlightedMessage + detectionStatus);
        // } else {
        //     detectionStatus = " (Status: Safe)";
        // }

        String displayMessage = "From: " + sender + "\n" + message + detectionStatus;
        smsMessages.add(0, displayMessage); // Add to the top of the list
        smsAdapter.notifyDataSetChanged();
        Log.d(TAG, "SMS received and displayed: " + displayMessage);
    }
}