package com.vectordb;

import java.util.*;

public class TextChunker {
    // Splits text into chunks of `size` words, each overlapping the previous by `overlap` words
    public static List<String> chunk(String text, int size, int overlap) {
        String[] words = text.trim().split("\\s+");
        if (words.length == 0) return List.of();
        if (words.length <= size) return List.of(text);

        List<String> chunks = new ArrayList<>();
        int step = size - overlap;
        for (int start = 0; start < words.length; start += step) {
            int end = Math.min(start + size, words.length);
            StringBuilder sb = new StringBuilder();
            for (int i = start; i < end; i++) {
                if (i > start) sb.append(' ');
                sb.append(words[i]);
            }
            chunks.add(sb.toString());
            if (end == words.length) break;
        }
        return chunks;
    }
}
