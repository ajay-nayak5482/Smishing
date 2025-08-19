package com.ajay.smishing;


import android.content.Context;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class BertTokenizer {
    private final Map<String, Integer> vocab;
    private final int maxSeqLen;

    public BertTokenizer(Context context, String vocabFileName, int maxLen) throws IOException {
        vocab = loadVocab(context, vocabFileName);
        maxSeqLen = maxLen;
    }

    private Map<String, Integer> loadVocab(Context context, String fileName) throws IOException {
        Map<String, Integer> vocab = new HashMap<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(context.getAssets().open(fileName)));
        String line;
        int index = 0;
        while ((line = reader.readLine()) != null) {
            vocab.put(line.trim(), index++);
        }
        reader.close();
        return vocab;
    }

    public int[] tokenize(String text) {
        List<Integer> tokenIds = new ArrayList<>();
        tokenIds.add(vocab.get("[CLS]"));

        for (String token : text.toLowerCase().split("\\s+")) {
            if (vocab.containsKey(token)) {
                tokenIds.add(vocab.get(token));
            } else {
                tokenIds.add(vocab.get("[UNK]"));
            }
        }

        tokenIds.add(vocab.get("[SEP]"));

        // Padding
        while (tokenIds.size() < maxSeqLen) tokenIds.add(0);
        if (tokenIds.size() > maxSeqLen) tokenIds = tokenIds.subList(0, maxSeqLen);

        // Convert to primitive array
        int[] ids = new int[maxSeqLen];
        for (int i = 0; i < maxSeqLen; i++) {
            ids[i] = tokenIds.get(i);
        }
        return ids;
    }
}

