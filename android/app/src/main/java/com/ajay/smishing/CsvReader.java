package com.ajay.smishing;

import android.content.Context;
import android.util.Log;

import com.opencsv.CSVReader;

import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class CsvReader {

    private static String TAG = "CsvReader";
    public static List<SmsData> readSmsData(Context context, String fileName) {
        List<SmsData> smsList = new ArrayList<>();

        try {
            InputStream is = context.getAssets().open(fileName);
            CSVReader reader = new CSVReader(new InputStreamReader(is));

            String[] nextLine;
            boolean isFirstLine = true;

            while ((nextLine = reader.readNext()) != null) {
                if (isFirstLine) {
                    isFirstLine = false; // skip header
                    continue;
                }

                if (nextLine.length >= 2) {
                    String mainText = nextLine[0];
                    String phishing = nextLine[1];
                    Log.d(TAG, "MainText: " + mainText + ", Phishing: " + phishing);

                    smsList.add(new SmsData(mainText, Integer.parseInt(phishing)));
                }
            }

            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
        }

        return smsList;
    }
}
