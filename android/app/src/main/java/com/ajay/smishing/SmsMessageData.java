package com.ajay.smishing;
import java.io.Serializable;

public class SmsMessageData implements Serializable {
    private String sender;
    private String body;
    private String date;
    private String detectionResult;
    private boolean isPhishing;
    private float probability;
    private final int isPhishingLabel; // 0 not used, 1 not phishing, 2 or beyond : phishing

    public SmsMessageData(String sender, String body, String date, String detectionResult,
                          boolean isPhishing, float probability, int isPhishingLabel) {
        this.sender = sender;
        this.body = body;
        this.date = date;
        this.detectionResult = detectionResult;
        this.isPhishing = isPhishing;
        this.probability = probability;
        this.isPhishingLabel = isPhishingLabel;
    }

    public String getSender() { return sender; }
    public String getBody() { return body; }
    public String getDate() { return date; }
    public String getDetectionResult() { return detectionResult; }
    public boolean isPhishing() { return isPhishing; }
    public float getProbability() { return probability; }

    public int getIsPhishingLabel() { return isPhishingLabel; }
}