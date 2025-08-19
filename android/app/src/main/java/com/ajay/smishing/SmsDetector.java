// This class encapsulates all ML model loading, preprocessing, and inference logic.
package com.ajay.smishing;

import android.content.Context;
import android.util.Log;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.nio.MappedByteBuffer;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

public class SmsDetector {

    public static final float THRESHOLD = 0.7f;
    private static final String TAG = "SmsDetector";
    private static final String MODEL_PATH = "final_robust_hybrid_model.tflite";
    private static final String VOCAB_PATH = "vocab.txt";
    private static final String SCALER_PARAMS_PATH = "scaler_params.json";
    private static final String ENCODER_PARAMS_PATH = "encoder_params.json";
    private static final String FEATURE_ORDER_PATH = "structured_feature_order.json";
    // --- Regex patterns and keywords (must match Python scripts) ---
    private static final Pattern URL_PATTERN = Pattern.compile("https?://\\S+|www\\.\\S+|[a-zA-Z0-9-]+\\.[a-zA-Z]{2,}");
    private static final Pattern EMAIL_PATTERN = Pattern.compile("\\S+@\\S+\\.\\S+");
    private static final Pattern PHONE_PATTERN = Pattern.compile("\\b\\d{10}\\b|\\(\\d{3}\\)\\s*\\d{3}-\\d{4}|\\d{3}[-.\\s]\\d{3}[-.\\s]\\d{4}");
    private static final List<String> SUSPICIOUS_TLDS = Arrays.asList("ru", "cn", "xyz", "top", "loan", "biz", "info", "online");
    private static final List<String> URGENT_KEYWORDS = Arrays.asList("urgent", "immediately", "action");
    // Simplified phishing keywords list from config.py
    private static final List<String> PHISHING_KEYWORDS = Arrays.asList(
            "account", "verify", "update", "security", "password", "login", "click",
            "link", "urgent", "alert", "suspicious", "bank", "credit", "card",
            "confirm", "fraud", "prize", "winner", "claim", "deliver", "tracking",
            "invoice", "payment", "transfer", "dear customer", "congratulations",
            "restricted", "action", "required", "immediately", "now", "blocked",
            "suspended", "compromised", "unusual", "activity", "verify", "confirm",
            "secure", "attention", "important", "warning", "error", "problem",
            "delivery", "parcel", "shipment", "order", "transaction", "invoice",
            "refund", "tax", "irs", "hmrc", "government", "fine", "penalty",
            "lottery", "winnings", "gift", "coupon", "free", "offer", "exclusive",
            "limited time", "expires", "congratulations", "selected", "eligible",
            "prize", "reward", "cash", "money", "payment", "transfer", "deposit",
            "withdraw", "loan", "credit", "debit", "card", "pin", "atm", "balance",
            "statement", "bill", "due", "overdue", "invoice", "receipt", "charge",
            "transaction", "purchase", "order", "shipping", "tracking", "delivery",
            "package", "shipment", "dispatch", "courier", "post", "mail",
            "support", "customer service", "help desk", "technical support",
            "service", "issue", "problem", "fix", "resolve", "restore", "recover",
            "reset", "reactivate", "unlock", "disable", "enable", "access",
            "personal", "information", "details", "data", "credentials", "identity",
            "ssn", "dob", "address", "phone", "email", "username", "password",
            "otp", "code", "token", "verification", "authentication", "authorization",
            "security", "fraud", "scam", "spam", "malicious", "suspicious", "unsafe",
            "warning", "alert", "notice", "notification", "message", "text", "sms",
            "call", "dial", "visit", "link", "website", "url", "portal", "page",
            "form", "survey", "update", "upgrade", "install", "download", "app",
            "software", "program", "virus", "malware", "trojan", "ransomware",
            "exploit", "vulnerability", "breach", "leak", "data breach",
            "urgent action required", "account suspended", "click here", "verify your account",
            "unusual login activity", "security alert", "prize winner", "claim your reward",
            "delivery failed", "track your package", "payment pending", "invoice attached",
            "tax refund", "government grant", "password reset", "confirm your identity",
            "your account has been locked", "update your details", "suspicious transaction",
            "call us now", "visit our website", "download the app", "important notice",
            "final warning", "immediate action", "security warning", "phishing detected",
            "spam alert", "malware detected", "fraudulent activity", "unauthorized access",
            "click the link", "login to your account", "bank alert", "credit card fraud",
            "urgent message", "delivery notification", "package tracking", "payment confirmation",
            "winning notification", "gift card", "free gift", "limited offer",
            "customer support", "technical issue", "account recovery", "password change",
            "security update", "data verification", "identity theft", "social security number",
            "date of birth", "personal data", "login credentials", "one-time password",
            "verification code", "authentication required", "authorization required",
            "security threat", "fraud alert", "scam alert", "spam warning", "malicious software",
            "suspicious activity", "unsafe link", "warning message", "alert notification",
            "text message", "call now", "dial number", "visit link", "website access",
            "portal login", "page access", "form submission", "survey link", "update software",
            "upgrade system", "install app", "download program", "virus alert", "malware detected",
            "trojan warning", "ransomware threat", "exploit vulnerability", "data breach detected"
    );
    private final int MAX_LEN = 192;
    private Interpreter tflite;
    private BertTokenizer tokenizer;
    private Context context;
    // Preprocessing parameters loaded from JSON
    private float[] scalerMean;
    private float[] scalerScale;
    private List<String> scalerFeatureNames;
    private List<List<String>> encoderCategories;
    private List<String> structuredFeatureOrder;

    public SmsDetector(Context context) throws IOException {
        this.context = context;
        loadModel();
        loadTokenizer();
        loadPreprocessingParams();
    }

    private void loadModel() throws IOException {
        MappedByteBuffer modelBuffer = FileUtil.loadMappedFile(context, MODEL_PATH);
        tflite = new Interpreter(modelBuffer, new Interpreter.Options());
        Log.d(TAG, "TFLite model loaded.");
    }

    private void loadTokenizer() throws IOException {
        try {
            tokenizer = new BertTokenizer(context, VOCAB_PATH, MAX_LEN);
        } catch (Exception e) {
            Log.e(TAG, "Error loading BertTokenizer from vocab.txt: " + e.getMessage());
            throw new IOException("Failed to load tokenizer from assets.");
        }

        Log.d(TAG, "Tokenizer loaded.");
    }

    private void loadPreprocessingParams() throws IOException {
        Gson gson = new Gson();

        String scalerJson = loadAssetJson(SCALER_PARAMS_PATH);
        Map<String, Object> scalerParamsMap = gson.fromJson(scalerJson, new TypeToken<Map<String, Object>>() {
        }.getType());
        scalerMean = listToFloatArray((List<Double>) scalerParamsMap.get("mean"));
        scalerScale = listToFloatArray((List<Double>) scalerParamsMap.get("scale"));
        scalerFeatureNames = (List<String>) scalerParamsMap.get("feature_names_in");
        Log.d(TAG, "Scaler params loaded for " + scalerFeatureNames.size() + " features.");

        String encoderJson = loadAssetJson(ENCODER_PARAMS_PATH);
        Map<String, Object> encoderParamsMap = gson.fromJson(encoderJson, new TypeToken<Map<String, Object>>() {
        }.getType());
        encoderCategories = (List<List<String>>) encoderParamsMap.get("categories");
        Log.d(TAG, "Encoder params loaded with " + encoderCategories.size() + " categories.");

        String orderJson = loadAssetJson(FEATURE_ORDER_PATH);
        structuredFeatureOrder = gson.fromJson(orderJson, new TypeToken<List<String>>() {
        }.getType());
        Log.d(TAG, "Structured feature order loaded. Total features: " + structuredFeatureOrder.size());
    }

    private String loadAssetJson(String filePath) throws IOException {
        InputStream is = context.getAssets().open(filePath);
        int size = is.available();
        byte[] buffer = new byte[size];
        is.read(buffer);
        is.close();
        return new String(buffer, "UTF-8");
    }

    private float[] listToFloatArray(List<Double> list) {
        float[] array = new float[list.size()];
        for (int i = 0; i < list.size(); i++) {
            array[i] = list.get(i).floatValue();
        }
        return array;
    }

    private String cleanTextForHeuristics(String text) {
        if (text == null) return "";
        text = text.toLowerCase();
        text = text.replaceAll("https?://\\S+|www\\.\\S+", "");
        text = text.replaceAll("\\[.*?\\]", "");
        text = text.replaceAll("<.*?>+", "");
        text = text.replaceAll("[\\p{Punct}]", "");
        text = text.replaceAll("\\d+", "");
        text = text.replaceAll("\\s+", " ").trim();
        return text;
    }

    private Map<String, Object> extractRawFeatures(String smsContent) {
        Map<String, Object> features = new LinkedHashMap<>();

        String cleanedText = cleanTextForHeuristics(smsContent);
        int smsLength = smsContent.length();
        int numSpecialChars = smsContent.replaceAll("[^\\p{Punct}]", "").length();
        int numDigits = smsContent.replaceAll("\\D", "").length();
        int numAllCapsWords = (int) Arrays.stream(cleanedText.split("\\s+"))
                .filter(word -> word.length() > 1 && word.matches("[A-Z]+")).count();
        int phishingKeywordsCount = (int) PHISHING_KEYWORDS.stream()
                .filter(keyword -> cleanedText.contains(keyword)).count();
        int wordCount = cleanedText.split("\\s+").length;
        float avgWordLength = (float) (wordCount > 0 ? (double) cleanedText.length() / wordCount : 0.0);

        features.put("sms_length", (float) smsLength);
        features.put("num_special_chars", (float) numSpecialChars);
        features.put("num_digits", (float) numDigits);
        features.put("num_all_caps_words", (float) numAllCapsWords);
        features.put("phishing_keywords_count", (float) phishingKeywordsCount);
        features.put("word_count", (float) wordCount);
        features.put("avg_word_length", avgWordLength);

        features.put("digits_to_length_ratio", (float) numDigits / (smsLength > 0 ? smsLength : 1e-6f));
        features.put("special_chars_to_length_ratio", (float) numSpecialChars / (smsLength > 0 ? smsLength : 1e-6f));
        features.put("phishing_keywords_to_word_count_ratio", (float) phishingKeywordsCount / (wordCount > 0 ? wordCount : 1e-6f));

        boolean hasUrl = URL_PATTERN.matcher(smsContent).find();
        boolean hasEmail = EMAIL_PATTERN.matcher(smsContent).find();
        boolean hasPhone = PHONE_PATTERN.matcher(smsContent).find();
        features.put("has_url", hasUrl ? 1.0f : 0.0f);
        features.put("has_email", hasEmail ? 1.0f : 0.0f);
        features.put("has_phone_number", hasPhone ? 1.0f : 0.0f);

        features.put("domain_age_days", 0.0f);
        features.put("is_ip_address_url", 0.0f);
        features.put("url_and_ip", 0.0f);
        features.put("url_and_suspicious_tld", 0.0f);
        features.put("urgent_and_url", (hasUrl && URGENT_KEYWORDS.stream().anyMatch(cleanedText::contains)) ? 1.0f : 0.0f);
        features.put("sentiment_neg", 0.0f);
        features.put("sentiment_neu", 0.0f);
        features.put("sentiment_pos", 0.0f);
        features.put("sentiment_compound", 0.0f);

        features.put("domain_encoded", 0.0f);
        features.put("tld_encoded", 0.0f);
        features.put("subdomain_encoded", 0.0f);
        features.put("url_subcategory_encoded", 0.0f);
        features.put("domain_registrar_encoded", 0.0f);
        features.put("sender_encoded", 0.0f);
        features.put("sender_type_encoded", 0.0f);
        features.put("brand_encoded", 0.0f);
        features.put("message_category_encoded", 0.0f);
        features.put("dataset_source_encoded", 0.0f);

        return features;
    }

    public float[] preprocessStructuredFeatures(Map<String, Object> rawFeatures) {
        float[] finalFeatures = new float[structuredFeatureOrder.size()];

        for (int i = 0; i < structuredFeatureOrder.size(); i++) {
            String featureName = structuredFeatureOrder.get(i);

            if (featureName.startsWith("num__") || featureName.startsWith("bool__")) {
                String originalFeatureName = featureName.replace("num__", "").replace("bool__", "");
                float rawValue = ((Number) rawFeatures.getOrDefault(originalFeatureName, 0.0f)).floatValue();

                int scalerIndex = scalerFeatureNames.indexOf(originalFeatureName);
                if (scalerIndex != -1) {
                    finalFeatures[i] = (rawValue - scalerMean[scalerIndex]) / (scalerScale[scalerIndex] != 0 ? scalerScale[scalerIndex] : 1e-6f);
                } else {
                    Log.w(TAG, "Scaler feature not found in map: " + originalFeatureName);
                    finalFeatures[i] = rawValue;
                }
            } else if (featureName.startsWith("cat__")) {
                String originalFeatureName = featureName.substring(5, featureName.lastIndexOf("_"));
                String categoryValue = featureName.substring(featureName.lastIndexOf("_") + 1);

                if (originalFeatureName.equals("dataset_source") && categoryValue.equals("D1")) {
                    finalFeatures[i] = 1.0f;
                } else {
                    finalFeatures[i] = 0.0f;
                }
            }
        }
        return finalFeatures;
    }

    public String detectPhishing(String smsContent) {
        // 1. Tokenize text input (input_ids, attention_mask)
        int[] tokenIds = tokenizer.tokenize(smsContent);
        int[] inputIds = new int[MAX_LEN];
        int[] attentionMask = new int[MAX_LEN];

        for (int i = 0; i < tokenIds.length && i < MAX_LEN; i++) {
            inputIds[i] = tokenIds[i];
            attentionMask[i] = 1;
        }

        // 2. Prepare structured features
        Map<String, Object> rawFeatures = extractRawFeatures(smsContent);
        float[] structuredFeatures = preprocessStructuredFeatures(rawFeatures);

        // 3. Run Inference
        try {
            // Create the input and attention mask buffers as FloatBuffer, not IntBuffer
            ByteBuffer inputIdsBuffer = ByteBuffer.allocateDirect(MAX_LEN * 4).order(ByteOrder.nativeOrder());
            IntBuffer inputIdsFloatBuffer = inputIdsBuffer.asIntBuffer();
            for (int id : inputIds) {
                inputIdsFloatBuffer.put(id);
            }
            inputIdsBuffer.rewind();

            ByteBuffer attentionMaskBuffer = ByteBuffer.allocateDirect(MAX_LEN * 4).order(ByteOrder.nativeOrder());
            IntBuffer attentionMaskFloatBuffer = attentionMaskBuffer.asIntBuffer();
            for (int mask : attentionMask) {
                attentionMaskFloatBuffer.put(mask);
            }
            attentionMaskBuffer.rewind();

            // Create structured features buffer
            ByteBuffer structuredFeaturesBuffer = ByteBuffer.allocateDirect(structuredFeatures.length * 4).order(ByteOrder.nativeOrder());
            structuredFeaturesBuffer.asFloatBuffer().put(structuredFeatures);
            structuredFeaturesBuffer.rewind();

            Object[] tfliteInputs = {attentionMaskBuffer, structuredFeaturesBuffer, inputIdsBuffer};

            float[][] output = new float[1][1];
            Map<Integer, Object> tfliteOutputs = new java.util.HashMap<>();
            tfliteOutputs.put(0, output);
            tflite.runForMultipleInputsOutputs(tfliteInputs, tfliteOutputs);


            float phishingProb = output[0][0];

            // 4. Post-processing
            String result;

            if (phishingProb >= THRESHOLD) {
                result = "PHISHING DETECTED!";
            } else {
                result = "Safe SMS.";
            }
            Log.d(TAG, " result: " + result + "Phishing Probability: " + phishingProb + " for SMS: " + smsContent);
            return String.format("%s (Probability: %.2f)", result, phishingProb);

        } catch (Exception e) {
            Log.e(TAG, "TFLite inference failed: " + e.getMessage());
            e.printStackTrace();
            return "Detection failed due to an error.";
        }
    }

    public void close() {
        if (tflite != null) {
            tflite.close();
        }
    }
}