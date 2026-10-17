package com.vectordb;

import java.util.*;
import java.util.function.BiFunction;

public class KDTree {
    private static class KDNode {
        VectorItem item;
        KDNode left, right;
        KDNode(VectorItem item) { this.item = item; }
    }

    private KDNode root;
    private final int dims;

    public KDTree(int dims) {
        this.dims = dims;
    }

    private KDNode ins(KDNode node, VectorItem item, int depth) {
        if (node == null) return new KDNode(item);
        int axis = depth % dims;
        if (item.emb.get(axis) < node.item.emb.get(axis))
            node.left = ins(node.left, item, depth + 1);
        else
            node.right = ins(node.right, item, depth + 1);
        return node;
    }

    public void insert(VectorItem item) {
        root = ins(root, item, 0);
    }

    // `best` is a max-heap of [distance, id] holding the k nearest found so far
    private void knn(KDNode node, List<Float> q, int k, int depth,
                     BiFunction<List<Float>, List<Float>, Float> dist,
                     PriorityQueue<float[]> best) {
        if (node == null) return;
        float d = dist.apply(q, node.item.emb);
        if (best.size() < k || d < best.peek()[0]) {
            best.offer(new float[]{d, node.item.id});
            if (best.size() > k) best.poll();
        }

        int axis = depth % dims;
        float diff = q.get(axis) - node.item.emb.get(axis);
        KDNode near = diff < 0 ? node.left : node.right;
        KDNode far  = diff < 0 ? node.right : node.left;
        knn(near, q, k, depth + 1, dist, best);
        // Only exact for Euclidean: other metrics may prune a branch that holds a closer point
        if (best.size() < k || Math.abs(diff) < best.peek()[0])
            knn(far, q, k, depth + 1, dist, best);
    }

    public List<float[]> knn(List<Float> q, int k,
                             BiFunction<List<Float>, List<Float>, Float> dist) {
        PriorityQueue<float[]> best = new PriorityQueue<>((a, b) -> Float.compare(b[0], a[0]));
        knn(root, q, k, 0, dist, best);
        List<float[]> res = new ArrayList<>(best);
        res.sort(Comparator.comparingDouble(a -> a[0]));
        return res;
    }

    public void rebuild(List<VectorItem> items) {
        root = null;
        for (VectorItem v : items) insert(v);
    }
}
