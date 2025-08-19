package com.ajay.smishing;

import android.os.Parcel;
import android.os.Parcelable;

import androidx.annotation.NonNull;

public class SmsData implements Parcelable {
    private final String mainText;
    private final int phishing; // could be boolean if desired

    public SmsData(String mainText, int phishing) {
        this.mainText = mainText;
        this.phishing = phishing;
    }

    protected SmsData(Parcel in) {
        mainText = in.readString();
        phishing = in.readInt();
    }

    public String getMainText() {
        return mainText;
    }

    public int getPhishing() {
        return phishing;
    }

    @NonNull
    @Override
    public String toString() {
        return "SmsData{" +
                "mainText='" + mainText + '\'' +
                ", phishing='" + phishing + '\'' +
                '}';
    }

    @Override
    public void writeToParcel(Parcel dest, int flags) {
        dest.writeString(mainText);
        dest.writeInt(phishing);
    }

    @Override
    public int describeContents() {
        return 0;
    }

    public static final Parcelable.Creator<SmsData> CREATOR = new Parcelable.Creator<SmsData>() {
        @Override
        public SmsData createFromParcel(Parcel in) {
            return new SmsData(in);
        }

        @Override
        public SmsData[] newArray(int size) {
            return new SmsData[size];
        }
    };
}
