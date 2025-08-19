// --- SmsReceiver.java ---
package com.ajay.smishing;
import static android.telephony.TelephonyManager.PHONE_TYPE_CDMA;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.provider.Telephony;
import android.telephony.SmsMessage;
import android.telephony.TelephonyManager;
import android.util.Log;
import android.widget.Toast;

import androidx.localbroadcastmanager.content.LocalBroadcastManager;

import java.io.IOException;

public class SmsReceiver extends BroadcastReceiver {

    private static final String TAG = "SmsReceiver";
    public static final String SMS_RECEIVED_ACTION = "com.ajay.smishing.SMS_RECEIVED";

    @Override
    public void onReceive(Context context, Intent intent) {
        if (intent.getAction() != null && intent.getAction().equals(Telephony.Sms.Intents.SMS_RECEIVED_ACTION)) {
            Bundle bundle = intent.getExtras();
            if (bundle != null) {
                Log.i(TAG, "bundle is not null");
                Object[] pdus = (Object[]) bundle.get("pdus");
                if (pdus != null) {
                    Log.i(TAG, "got PDUS");
                    StringBuilder smsBody = new StringBuilder();
                    String sender = "";

                    int count = 1;
                    for (Object pdu : pdus) {
                        TelephonyManager telephonyManager = (TelephonyManager) context.getSystemService(Context.TELEPHONY_SERVICE);
                        int activePhone =telephonyManager.getPhoneType();
                        String format = (PHONE_TYPE_CDMA == activePhone) ?
                                SmsMessage.FORMAT_3GPP2 : SmsMessage.FORMAT_3GPP;
                        SmsMessage smsMessage = SmsMessage.createFromPdu((byte[]) pdu, format);
                        sender = smsMessage.getDisplayOriginatingAddress();
                        smsBody.append(smsMessage.getMessageBody());
                        Log.d(TAG, "found pdu "+count++);
                    }

                    String fullSmsMessage = smsBody.toString();
                    Log.d(TAG, "SMS received - Sender: " + sender + ", Message: " + fullSmsMessage);

                    // Create a new Intent for local broadcast
                    Intent localIntent = new Intent(SMS_RECEIVED_ACTION);
                    localIntent.putExtra("sender", sender);
                    localIntent.putExtra("message", fullSmsMessage);

                    // Send the local broadcast
                    LocalBroadcastManager.getInstance(context).sendBroadcast(localIntent);


//                    if (context instanceof MainActivity) {
//                        ((MainActivity) context).updateSmsDisplay(sender, fullSmsMessage);
//                    } else {
//                        Log.w(TAG, "MainActivity not active to display SMS. Attempting detection anyway.");
//                        try {
//                            SmsDetector tempDetector = new SmsDetector(context);
//                            //tempDetector.initialize();
//                            String result = tempDetector.detectPhishing(fullSmsMessage);
//                            tempDetector.close();
//                            Log.i(TAG, "Background detection result: " + result);
//                            Toast.makeText(context, "SMS from " + sender + " detected as: " + result, Toast.LENGTH_LONG).show();
//                        } catch (IOException e) {
//                            Log.e(TAG, "Error during background detection: " + e.getMessage());
//                        }
//                    }
                }
            }
        }
    }
}