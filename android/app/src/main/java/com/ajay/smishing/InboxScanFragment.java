// InboxScanFragment.java
package com.ajay.smishing;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ListView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.fragment.app.Fragment;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.Locale;

public class InboxScanFragment extends Fragment implements OnSmsReceivedListener {

    private static final String TAG = "InboxScanFragment";
    private LinearLayout loadingLayout;

    private SmsAdapter smsAdapter;
    private ArrayList<SmsMessageData> smsMessages;
    // Get the SmsDetector from the parent activity
    private SmsDetector smsDetector = null;
    private volatile boolean stopScan = false;
    private Thread scanThread = null;
    private TextView smsCount = null;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_inbox_scan, container, false);

        ListView smsListView = view.findViewById(R.id.smsListView);
        loadingLayout = view.findViewById(R.id.loadingLayout);
        Button cancelButton = view.findViewById(R.id.cancelButton);
        smsCount = view.findViewById(R.id.smsCount);

        smsMessages = new ArrayList<>();
        smsAdapter = new SmsAdapter(view.getContext(), smsMessages);
        smsListView.setAdapter(smsAdapter);
        smsListView.setEmptyView(view.findViewById(R.id.empty_view));

        smsDetector = ((MainActivity) requireActivity()).getSmsDetector();
        // Set up item click listener to open DetailActivity
        smsListView.setOnItemClickListener((parent, view1, position, id) -> {
            SmsMessageData clickedItem = (SmsMessageData) parent.getItemAtPosition(position);

            Intent intent = new Intent(view.getContext(), DetailActivity.class);
            intent.putExtra("sender", clickedItem.getSender());
            intent.putExtra("body", clickedItem.getBody());
            intent.putExtra("date", clickedItem.getDate());
            intent.putExtra("detection_result", clickedItem.getDetectionResult());
            intent.putExtra("is_phishing", clickedItem.isPhishing());
            intent.putExtra("probability", clickedItem.getProbability());
            startActivity(intent);
        });

        cancelButton.setOnClickListener(v -> {
            stopScan = true;
            Toast.makeText(getContext(), "Scan cancelled.", Toast.LENGTH_SHORT).show();
            loadingLayout.setVisibility(View.GONE);
        });

        if (smsMessages == null || smsMessages.isEmpty()) {
            showSMSReadConfirmation(getContext());
        }
        return view;
    }

    private void readExistingSmsInBackground() {
        if (scanThread != null && scanThread.isAlive()) {
            return; // Don't start a new scan if one is already running
        }

        stopScan = false;
        loadingLayout.setVisibility(View.VISIBLE);

        scanThread = new Thread(this::readExistingSms);
        scanThread.start();
    }

    @SuppressLint("DefaultLocale")
    private void readExistingSms() {
        Uri uri = Uri.parse("content://sms/inbox");
        String[] projection = {"_id", "address", "body", "date"};

        try (Cursor cursor = requireContext().getContentResolver().query(uri, projection, null, null, "date DESC")) {
            if (cursor != null && cursor.moveToFirst()) {
                smsMessages.clear();
                do {

                    if (stopScan) {
                        Log.i(TAG, "SMS scan cancelled by user.");
                        break; // Exit the loop gracefully
                    }

                    String sender = cursor.getString(cursor.getColumnIndexOrThrow("address"));
                    String messageBody = cursor.getString(cursor.getColumnIndexOrThrow("body"));
                    long messageDate = cursor.getLong(cursor.getColumnIndexOrThrow("date"));

                    // Perform detection
                    String detectionResult = smsDetector.detectPhishing(messageBody);
                    boolean isPhishing = detectionResult.contains("PHISHING");
                    String probabilityString = detectionResult.substring(detectionResult.indexOf(":") + 2, detectionResult.indexOf(")"));
                    float probability = Float.parseFloat(probabilityString);

                    SimpleDateFormat sdf = new SimpleDateFormat("MMM dd, yyyy HH:mm", Locale.getDefault());
                    String formattedDate = sdf.format(new Date(messageDate));

                    // Add to our list of custom data objects
                    smsMessages.add(new SmsMessageData(sender, messageBody, formattedDate, isPhishing ? "SMISHING Detected":"Safe SMS", isPhishing, probability, 0));
                    requireActivity().runOnUiThread(() -> {
                        smsAdapter.notifyDataSetChanged();
                        smsCount.setText(String.format("%s : %d", getString(R.string.processed_sms_s), smsMessages.size()));
                    });

                } while (cursor.moveToNext());

                Log.d(TAG, "Finished reading and categorizing " + smsMessages.size() + " existing SMS messages.");
            } else {
                Log.w(TAG, "No existing SMS messages found.");
            }
        } catch (SecurityException e) {
            Log.e(TAG, "Permission to read SMS was denied.", e);
            requireActivity().runOnUiThread(() -> Toast.makeText(getContext(), "Permission denied. Cannot read existing SMS.", Toast.LENGTH_LONG).show());
        }
    }

    // This method is called by SmsReceiver for new messages
    public void updateSmsDisplay(String sender, String message) {
        updateMessageWithDetection(sender, message);
    }

    private void updateMessageWithDetection(String sender, String message) {
        String detectionResult = smsDetector.detectPhishing(message);
        boolean isPhishing = detectionResult.contains("PHISHING");
        String probabilityString = detectionResult.substring(detectionResult.indexOf(":") + 2, detectionResult.indexOf(")"));
        float probability = Float.parseFloat(probabilityString);

        SimpleDateFormat sdf = new SimpleDateFormat("MMM dd, yyyy HH:mm", Locale.getDefault());
        String formattedDate = sdf.format(new Date());

        SmsMessageData newSms = new SmsMessageData(sender, message, formattedDate, isPhishing ? "SMISHING Detected":"Safe SMS", isPhishing, probability, 0);

        requireActivity().runOnUiThread(() -> {
            smsMessages.add(0, newSms);
            smsAdapter.notifyDataSetChanged();
            Log.d(TAG, "New SMS received and displayed: " + detectionResult);
        });
    }

    private void showSMSReadConfirmation(Context context) {
        new AlertDialog.Builder(context).setTitle("Permissions for SMS Analysis")
                .setMessage("This app can classify your existing SMS messages to provide a full analysis. Granting SMS access will enable this feature." +
                        "\nYou may proceed without granting this permission to classify new messages exclusively.")
                .setPositiveButton("Grant Access", (dialog, which) -> {
                    dialog.dismiss();
                    readExistingSmsInBackground();
                })
                .setNegativeButton("No Thanks", (dialog, which) -> {
                    dialog.dismiss();
                })
                .setIcon(R.drawable.no_content_image)
                .show();
    }

    @Override
    public void onSmsReceived(String sender, String message) {
        updateSmsDisplay(sender, message);
    }
}
