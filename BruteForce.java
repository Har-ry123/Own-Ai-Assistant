package com.vectordb;

import java.util.*;
import java.util.function.BiFunction;

public class BruteForce {
    public List<VectorItem> items = new ArrayList<>();

    public void insert(VectorItem item) {
        items.add(item);
    }

    // Returns up to k [distance, id] pairs, nearest first
    public List<float[]> knn(List<Float> q, int k,
                             BiFunction<List<Float>, List<Float>, Float> dist) {
        List<float[]> all = new ArrayList<>();
        for (VectorItem v : items)
            all.add(new float[]{dist.apply(q, v.emb), v.id});
        all.sort(Comparator.comparingDouble(a -> a[0]));
        return all.size() > k ? all.subList(0, k) : all;
    }

    public void remove(int id) {
        items.removeIf(v -> v.id == id);
    }
}
