// --- SmsReceiver.java ---
package com.ajay.smishing;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.telephony.SmsMessage;
import android.util.Log;

public class SmsReceiver extends BroadcastReceiver {

    private static final String TAG = "SmsReceiver";

    @Override
    public void onReceive(Context context, Intent intent) {
        if (intent.getAction() != null && intent.getAction().equals("android.provider.Telephony.SMS_RECEIVED")) {
            Bundle bundle = intent.getExtras();
            if (bundle != null) {
                Object[] pdus = (Object[]) bundle.get("pdus");
                if (pdus != null) {
                    StringBuilder smsBody = new StringBuilder();
                    String sender = "";

                    for (Object pdu : pdus) {
                        SmsMessage smsMessage = SmsMessage.createFromPdu((byte[]) pdu);
                        sender = smsMessage.getDisplayOriginatingAddress();
                        smsBody.append(smsMessage.getMessageBody());
                    }

                    String fullSmsMessage = smsBody.toString();
                    Log.d(TAG, "SMS received - Sender: " + sender + ", Message: " + fullSmsMessage);

                    // Pass the received SMS to MainActivity for display and processing
                    if (context instanceof MainActivity) {
                        ((MainActivity) context).updateSmsDisplay(sender, fullSmsMessage);
                    } else {
                        // If MainActivity is not active, you might want to start a service
                        // or store the SMS for later display. For this basic example,
                        // we'll just log it.
                        Log.w(TAG, "MainActivity not active. SMS not immediately displayed.");
                        // You could also show a toast for debugging:
                        // Toast.makeText(context, "New SMS from " + sender + ": " + fullSmsMessage, Toast.LENGTH_LONG).show();
                    }
                }
            }
        }
    }
}