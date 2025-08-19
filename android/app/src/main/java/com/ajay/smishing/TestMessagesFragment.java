// TestMessagesFragment.java
package com.ajay.smishing;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ListView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class TestMessagesFragment extends Fragment {

    private static final String TAG = "TestMessages";
    private SmsAdapter smsAdapter;
    private ArrayList<SmsMessageData> smsMessages;
    private SmsDetector smsDetector;
    private final Executor backgroundExecutor = Executors.newSingleThreadExecutor();

    // Handler to post results back to the main UI thread
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private List<SmsData> smsList = null;
    private LinearLayout loadingLayout;
    private volatile boolean stopScan = false;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_test_messages, container, false);
        ListView testListView = view.findViewById(R.id.testListView);
        smsMessages = new ArrayList<>();
        smsAdapter = new SmsAdapter(getContext(), smsMessages);
        testListView.setAdapter(smsAdapter);
        testListView.setEmptyView(view.findViewById(R.id.empty_view));
        loadingLayout = view.findViewById(R.id.loadingLayout);
        Button cancelButton = view.findViewById(R.id.cancelButton);

        smsDetector = ((MainActivity) getActivity()).getSmsDetector();
        startBackgroundTask();

        testListView.setOnItemClickListener((parent, view1, position, id) -> {
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


        return view;
    }

    private void runTests() throws InterruptedException {
        // Clear previous test results
        smsMessages.clear();
        int count = 1;
        for(SmsData sms : smsList) {
            if(stopScan)
                break;
            // Process each test message
            updateMessageWithDetection(String.format(Locale.getDefault(), "Row-%02d", count++), sms.getMainText(), sms.getPhishing()+1);

        }
    }

    private void updateMessageWithDetection(String sender, String message, int isPhishingLabel) {
        String detectionResult = smsDetector.detectPhishing(message);
        boolean isPhishing = detectionResult.contains("PHISHING");
        String probabilityString = detectionResult.substring(detectionResult.indexOf(":") + 2, detectionResult.indexOf(")"));
        float probability = Float.parseFloat(probabilityString);

        SimpleDateFormat sdf = new SimpleDateFormat("MMM dd, yyyy HH:mm", Locale.getDefault());
        String formattedDate = sdf.format(new Date());

        SmsMessageData newSms = new SmsMessageData(sender, message, formattedDate, isPhishing ? "SMISHING Detected":"Safe SMS", isPhishing, probability, isPhishingLabel);
        smsMessages.add(newSms);
        requireActivity().runOnUiThread(() -> {
            smsAdapter.notifyDataSetChanged();
        });
    }

    private void startBackgroundTask() {
        // Run the main logic after a short delay on the main UI thread
        mainHandler.postDelayed(() -> {
            // This code runs after the delay on the main UI thread

            // Execute the heavy task on a background thread
            backgroundExecutor.execute(this::doBackgroundTask);
        }, 500); // 500ms delay for splash
    }

    private void doBackgroundTask() {
        // This is the heavy lifting part of the code
        // It runs on a background thread
        try {

            requireActivity().runOnUiThread(()->{
                stopScan = false;
                loadingLayout.setVisibility(View.VISIBLE);
            });

            if(smsList == null) {
                smsList = CsvReader.readSmsData(getContext(), "test_data.csv");
            }
            runTests();
        } catch (InterruptedException e){
            Log.e(TAG, "Interrupted while running tests: " + e.getMessage());

        } finally {
            // After the background work is done, post the result back to the main UI thread
            mainHandler.post(this::onTaskCompleted);
        }
    }

    private void onTaskCompleted() {
        // This code runs on the main UI thread
        // Transition to the main activity
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        // It's a good practice to clean up any pending callbacks to prevent leaks
        mainHandler.removeCallbacksAndMessages(null);
    }
}
