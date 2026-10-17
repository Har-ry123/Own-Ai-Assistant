package com.vectordb;

import java.util.*;
import java.util.AbstractMap.SimpleEntry;
import java.util.concurrent.locks.ReentrantLock;

// Stores document chunks with their real (Ollama) embeddings for RAG
public class DocumentDB {
    public static class DocItem {
        public int id;
        public String title, text;
        public List<Float> emb;

        public DocItem(int id, String title, String text, List<Float> emb) {
            this.id = id;
            this.title = title;
            this.text = text;
            this.emb = emb;
        }
    }

    private final Map<Integer, DocItem> store = new HashMap<>();
    private final HNSW hnsw = new HNSW(16, 200);
    private final BruteForce bf = new BruteForce();
    private final ReentrantLock lock = new ReentrantLock();
    private int nextId = 1;
    private int dims = 0;

    public int insert(String title, String text, List<Float> emb) {
        lock.lock();
        try {
            if (dims == 0) dims = emb.size();
            DocItem d = new DocItem(nextId++, title, text, emb);
            store.put(d.id, d);
            VectorItem v = new VectorItem(d.id, title, "doc", emb);
            hnsw.insert(v, DistanceMetrics::cosine);
            bf.insert(v);
            return d.id;
        } finally { lock.unlock(); }
    }

    // Brute force is exact and fast enough for small collections; HNSW beyond that
    public List<float[]> searchRaw(List<Float> q, int k) {
        lock.lock();
        try {
            if (store.isEmpty()) return Collections.emptyList();
            return store.size() < 10
                    ? bf.knn(q, k, DistanceMetrics::cosine)
                    : hnsw.knn(q, k, 50, DistanceMetrics::cosine);
        } finally { lock.unlock(); }
    }

    // Like searchRaw, but drops hits farther than maxDist
    public List<SimpleEntry<Float, DocItem>> search(List<Float> q, int k, float maxDist) {
        List<float[]> raw = searchRaw(q, k);
        List<SimpleEntry<Float, DocItem>> hits = new ArrayList<>();
        lock.lock();
        try {
            for (float[] r : raw) {
                int id = (int) r[1];
                if (store.containsKey(id) && r[0] <= maxDist)
                    hits.add(new SimpleEntry<>(r[0], store.get(id)));
            }
        } finally { lock.unlock(); }
        return hits;
    }

    public boolean remove(int id) {
        lock.lock();
        try {
            if (!store.containsKey(id)) return false;
            store.remove(id);
            hnsw.remove(id);
            bf.remove(id);
            return true;
        } finally { lock.unlock(); }
    }

    public List<DocItem> all() {
        lock.lock();
        try { return new ArrayList<>(store.values()); }
        finally { lock.unlock(); }
    }

    public int size() {
        lock.lock();
        try { return store.size(); }
        finally { lock.unlock(); }
    }

    public int getDims() { return dims; }
}
