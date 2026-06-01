package com.vectordb;

import java.util.*;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.BiFunction;

public class VectorDB {
    private final Map<Integer, VectorItem> store = new HashMap<>();
    private final BruteForce bf = new BruteForce();
    private final KDTree kdt;
    private final HNSW hnsw;
    private final ReentrantLock lock = new ReentrantLock();
    private int nextId = 1;
    public final int dims;

    public VectorDB(int d) {
        this.dims = d;
        this.kdt = new KDTree(d);
        this.hnsw = new HNSW(16, 200);
    }

    public int insert(String meta, String cat, List<Float> emb,
                      BiFunction<List<Float>, List<Float>, Float> dist) {
        lock.lock();
        try {
            VectorItem v = new VectorItem(nextId++, meta, cat, emb);
            store.put(v.id, v);
            bf.insert(v);
            kdt.insert(v);
            hnsw.insert(v, dist);
            return v.id;
        } finally { lock.unlock(); }
    }

    public boolean remove(int id) {
        lock.lock();
        try {
            if (!store.containsKey(id)) return false;
            store.remove(id);
            bf.remove(id);
            hnsw.remove(id);
            kdt.rebuild(new ArrayList<>(store.values()));
            return true;
        } finally { lock.unlock(); }
    }

    public record Hit(int id, String meta, String cat, List<Float> emb, float dist) {}
    public record SearchOut(List<Hit> hits, long us, String algo, String metric) {}
    public record BenchOut(long bfUs, long kdUs, long hnswUs, int n) {}

    public SearchOut search(List<Float> q, int k, String metric, String algo) {
        lock.lock();
        try {
            var dfn = DistanceMetrics.getDistFn(metric);
            long t0 = System.nanoTime();
            List<float[]> raw = switch (algo) {
                case "bruteforce" -> bf.knn(q, k, dfn);
                case "kdtree"     -> kdt.knn(q, k, dfn);
                default           -> hnsw.knn(q, k, 50, dfn);
            };
            long us = (System.nanoTime() - t0) / 1000;
            List<Hit> hits = new ArrayList<>();
            for (float[] r : raw) {
                int id = (int) r[1];
                if (store.containsKey(id))
                    hits.add(new Hit(id, store.get(id).metadata, store.get(id).category, store.get(id).emb, r[0]));
            }
            return new SearchOut(hits, us, algo, metric);
        } finally { lock.unlock(); }
    }

    public BenchOut benchmark(List<Float> q, int k, String metric) {
        lock.lock();
        try {
            var dfn = DistanceMetrics.getDistFn(metric);
            long t;
            t = System.nanoTime(); bf.knn(q, k, dfn);  long bfUs  = (System.nanoTime() - t) / 1000;
            t = System.nanoTime(); kdt.knn(q, k, dfn); long kdUs  = (System.nanoTime() - t) / 1000;
            t = System.nanoTime(); hnsw.knn(q, k, 50, dfn); long hnswUs = (System.nanoTime() - t) / 1000;
            return new BenchOut(bfUs, kdUs, hnswUs, store.size());
        } finally { lock.unlock(); }
    }

    public List<VectorItem> all() {
        lock.lock();
        try { return new ArrayList<>(store.values()); }
        finally { lock.unlock(); }
    }

    public HNSW.GraphInfo hnswInfo() {
        lock.lock();
        try { return hnsw.getInfo(); }
        finally { lock.unlock(); }
    }

    public int size() {
        lock.lock();
        try { return store.size(); }
        finally { lock.unlock(); }
    }
}
