package com.ajay.smishing;

import android.Manifest;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.provider.Telephony;
import android.util.Log;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentManager;
import androidx.lifecycle.Lifecycle;
import androidx.viewpager2.adapter.FragmentStateAdapter;
import androidx.viewpager2.widget.ViewPager2;

import com.google.android.material.tabs.TabLayout;
import com.google.android.material.tabs.TabLayoutMediator;

import java.io.IOException;
import java.util.Objects;
import androidx.localbroadcastmanager.content.LocalBroadcastManager;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "PhishingDetector";
    private static final int SMS_PERMISSION_CODE = 100;

    private SmsReceiver smsReceiver;
    private SmsDetector smsDetector;
    private OnSmsReceivedListener smsListener;
    private BroadcastReceiver localSmsReceiver; // Receiver for local broadcasts

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        try {
            smsDetector = new SmsDetector(getApplicationContext());
        } catch (IOException e) {
            Log.e(TAG, "Error initializing SmsDetector: " + e.getMessage());
            Toast.makeText(this, "Error loading ML model: " + e.getMessage(), Toast.LENGTH_LONG).show();
            finish();
            return;
        }

        if (checkSmsPermissions()) {
            registerSmsReceiver();
        } else {
            requestSmsPermissions();
        }
    }

    private boolean checkSmsPermissions() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.RECEIVE_SMS) == PackageManager.PERMISSION_GRANTED
                && ContextCompat.checkSelfPermission(this, Manifest.permission.READ_SMS) == PackageManager.PERMISSION_GRANTED;
    }

    private void requestSmsPermissions() {
        ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_SMS,
                Manifest.permission.RECEIVE_SMS}, SMS_PERMISSION_CODE);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == SMS_PERMISSION_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED && grantResults[1] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "SMS permissions granted!", Toast.LENGTH_SHORT).show();
                registerSmsReceiver();
            } else {
                Toast.makeText(this, "SMS permissions denied. App may not function correctly.", Toast.LENGTH_LONG).show();
                finish();
            }
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    private void registerSmsReceiver() {

        TabLayout tabLayout = findViewById(R.id.tabLayout);
        ViewPager2 viewPager = findViewById(R.id.viewPager);

        viewPager.setAdapter(new ViewPagerAdapter(getSupportFragmentManager(), getLifecycle()));

        new TabLayoutMediator(tabLayout, viewPager, (tab, position) -> {
            if (position == 0) {
                tab.setText("Inbox");
            } else {
                tab.setText("Test Messages");
            }
        }).attach();

        for (int i = 0; i < tabLayout.getTabCount(); i++) {
            TabLayout.Tab tab = tabLayout.getTabAt(i);
            if (tab != null && tab.view != null) {
                tab.view.setBackgroundResource(R.drawable.tab_unselected_bg);
            }
        }

        tabLayout.getTabAt(0).view.setBackgroundResource(R.drawable.tab_selected_bg);

        tabLayout.addOnTabSelectedListener(new TabLayout.OnTabSelectedListener() {
            @Override
            public void onTabSelected(TabLayout.Tab tab) {
                tab.view.setBackgroundResource(R.drawable.tab_selected_bg);
            }

            @Override
            public void onTabUnselected(TabLayout.Tab tab) {
                tab.view.setBackgroundResource(R.drawable.tab_unselected_bg);
            }

            @Override
            public void onTabReselected(TabLayout.Tab tab) { }
        });

        if(smsReceiver != null) {
            smsReceiver = new SmsReceiver();
        }
        IntentFilter filter = new IntentFilter(Telephony.Sms.Intents.SMS_RECEIVED_ACTION);
        registerReceiver(smsReceiver, filter);
        Log.d(TAG, "SMS Receiver registered.");

        // Register the local broadcast receiver
        setupLocalBroadcastReceiver();
    }

    private void setupLocalBroadcastReceiver() {
        if(localSmsReceiver == null) {
            localSmsReceiver = new BroadcastReceiver() {
                @Override
                public void onReceive(Context context, Intent intent) {
                    String sender = intent.getStringExtra("sender");
                    String message = intent.getStringExtra("message");
                    updateSmsDisplay(sender, message);
                }
            };
        }
        LocalBroadcastManager.getInstance(this).registerReceiver(localSmsReceiver, new IntentFilter(SmsReceiver.SMS_RECEIVED_ACTION));
        Log.d(TAG, "Local SMS Receiver registered.");
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        // Unregister the local receiver when the activity is paused
        LocalBroadcastManager.getInstance(this).unregisterReceiver(localSmsReceiver);
        if (smsReceiver != null) {
            unregisterReceiver(smsReceiver);
            Log.d(TAG, "SMS Receiver unregistered.");
        }
        if (smsDetector != null) {
            smsDetector.close();
        }
    }

    // Public method to get the SmsDetector instance for Fragments
    public SmsDetector getSmsDetector() {
        return smsDetector;
    }

    public void updateSmsDisplay(String sender, String fullSmsMessage) {
        if (smsListener != null) {
            smsListener.onSmsReceived(sender, fullSmsMessage);
        }
    }

    // Public method for Fragments to register as listeners
    public void setSmsListener(OnSmsReceivedListener listener) {
        this.smsListener = listener;
    }

    // Adapter for the ViewPager
    private class ViewPagerAdapter extends FragmentStateAdapter {
        private static final int NUM_TABS = 2;

        public ViewPagerAdapter(@NonNull FragmentManager fragmentManager, @NonNull Lifecycle lifecycle) {
            super(fragmentManager, lifecycle);
        }

        @NonNull
        @Override
        public Fragment createFragment(int position) {
            Fragment fragment;
            if (position == 0) {
                fragment = new InboxScanFragment();
                // Pass the listener to the fragment (or set it)
                if (fragment instanceof InboxScanFragment) {
                    setSmsListener((InboxScanFragment) fragment);
                }
            } else {
                fragment = new TestMessagesFragment();
            }
            return fragment;
        }

        @Override
        public int getItemCount() {
            return NUM_TABS;
        }
    }
}