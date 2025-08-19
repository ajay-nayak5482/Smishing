package com.ajay.smishing;

import android.os.Bundle;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.ActionBar;
import androidx.appcompat.app.AppCompatActivity;

import android.view.MenuItem;

public class DetailActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_detail);

        // Get data from the Intent
        String sender = getIntent().getStringExtra("sender");
        String body = getIntent().getStringExtra("body");
        String date = getIntent().getStringExtra("date");
        String detectionResult = getIntent().getStringExtra("detection_result");
        boolean isPhishing = getIntent().getBooleanExtra("is_phishing", false);
        float probability = getIntent().getFloatExtra("probability", 0.0f);

        // Find views
        TextView senderTextView = findViewById(R.id.detailSenderTextView);
        TextView dateTextView = findViewById(R.id.detailDateTextView);
        TextView bodyTextView = findViewById(R.id.detailBodyTextView);
        TextView resultTextView = findViewById(R.id.detailResultTextView);
        TextView probabilityTextView = findViewById(R.id.detailProbabilityTextView);

        // Populate views
        senderTextView.setText(sender);
        dateTextView.setText(date);
        bodyTextView.setText(body);
        resultTextView.setText(detectionResult);
        probabilityTextView.setText(String.format("Phishing Probability: %.2f", probability));

        // Set color for the result text
        if (isPhishing) {
            resultTextView.setTextColor(getResources().getColor(R.color.red));
        } else {
            resultTextView.setTextColor(getResources().getColor(R.color.green));
        }

        ActionBar actionBar = getSupportActionBar();
        if(actionBar != null){
            actionBar.setDisplayHomeAsUpEnabled(true);
        }
    }

    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        if (item.getItemId() == android.R.id.home) {
            finish(); // or call super.onBackPressed()
            return true;
        }
        return super.onOptionsItemSelected((android.view.MenuItem) item);
    }
}
