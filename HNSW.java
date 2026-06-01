package com.vectordb;

import java.util.*;
import java.util.function.BiFunction;

public class HNSW {
    public static class GraphInfo {
        public int topLayer, nodeCount;
        public List<Integer> nodesPerLayer = new ArrayList<>();
        public List<Integer> edgesPerLayer = new ArrayList<>();
        public List<NodeView> nodes = new ArrayList<>();
        public List<EdgeView> edges = new ArrayList<>();

        public static class NodeView {
            public int id, maxLyr;
            public String metadata, category;
        }
        public static class EdgeView {
            public int src, dst, lyr;
        }
    }

    private static class Node {
        VectorItem item;
        int maxLyr;
        List<List<Integer>> nbrs;
        Node(VectorItem item, int maxLyr) {
            this.item = item;
            this.maxLyr = maxLyr;
            this.nbrs = new ArrayList<>();
            for (int i = 0; i <= maxLyr; i++) nbrs.add(new ArrayList<>());
        }
    }

    private final Map<Integer, Node> G = new HashMap<>();
    private final int M, M0, efBuild;
    private final float mL;
    private int topLayer = -1;
    private int entryPt = -1;
    private final Random rng = new Random(42);

    public HNSW(int m, int efBuild) {
        this.M = m;
        this.M0 = 2 * m;
        this.efBuild = efBuild;
        this.mL = 1.0f / (float) Math.log(m);
    }

    private int randLevel() {
        return (int) Math.floor(-Math.log(rng.nextFloat()) * mL);
    }

    private List<float[]> searchLayer(List<Float> q, int ep, int ef, int lyr,
                                       BiFunction<List<Float>, List<Float>, Float> dist) {
        Map<Integer, Boolean> vis = new HashMap<>();
        PriorityQueue<float[]> cands = new PriorityQueue<>(Comparator.comparingDouble(a -> a[0]));
        PriorityQueue<float[]> found = new PriorityQueue<>((a, b) -> Float.compare(b[0], a[0]));

        float d0 = dist.apply(q, G.get(ep).item.emb);
        vis.put(ep, true);
        cands.offer(new float[]{d0, ep});
        found.offer(new float[]{d0, ep});

        while (!cands.isEmpty()) {
            float[] top = cands.poll();
            float cd = top[0]; int cid = (int) top[1];
            if (found.size() >= ef && cd > found.peek()[0]) break;
            Node cn = G.get(cid);
            if (cn == null || lyr >= cn.nbrs.size()) continue;
            for (int nid : cn.nbrs.get(lyr)) {
                if (vis.getOrDefault(nid, false) || !G.containsKey(nid)) continue;
                vis.put(nid, true);
                float nd = dist.apply(q, G.get(nid).item.emb);
                if (found.size() < ef || nd < found.peek()[0]) {
                    cands.offer(new float[]{nd, nid});
                    found.offer(new float[]{nd, nid});
                    if (found.size() > ef) found.poll();
                }
            }
        }

        List<float[]> res = new ArrayList<>(found);
        res.sort(Comparator.comparingDouble(a -> a[0]));
        return res;
    }

    public void insert(VectorItem item, BiFunction<List<Float>, List<Float>, Float> dist) {
        int id = item.id;
        int lvl = randLevel();
        G.put(id, new Node(item, lvl));

        if (entryPt == -1) {
            entryPt = id;
            topLayer = lvl;
            return;
        }

        int ep = entryPt;
        for (int lc = topLayer; lc > lvl; lc--) {
            Node epNode = G.get(ep);
            if (epNode != null && lc < epNode.nbrs.size()) {
                List<float[]> W = searchLayer(item.emb, ep, 1, lc, dist);
                if (!W.isEmpty()) ep = (int) W.get(0)[1];
            }
        }

        for (int lc = Math.min(topLayer, lvl); lc >= 0; lc--) {
            List<float[]> W = searchLayer(item.emb, ep, efBuild, lc, dist);
            int maxM = (lc == 0) ? M0 : M;
            List<Integer> sel = new ArrayList<>();
            for (int i = 0; i < Math.min(W.size(), maxM); i++)
                sel.add((int) W.get(i)[1]);

            Node curNode = G.get(id);
            while (curNode.nbrs.size() <= lc) curNode.nbrs.add(new ArrayList<>());
            curNode.nbrs.set(lc, new ArrayList<>(sel));

            for (int nid : sel) {
                Node nNode = G.get(nid);
                if (nNode == null) continue;
                while (nNode.nbrs.size() <= lc) nNode.nbrs.add(new ArrayList<>());
                nNode.nbrs.get(lc).add(id);
                if (nNode.nbrs.get(lc).size() > maxM) {
                    List<float[]> ds = new ArrayList<>();
                    for (int c : nNode.nbrs.get(lc))
                        if (G.containsKey(c))
                            ds.add(new float[]{dist.apply(nNode.item.emb, G.get(c).item.emb), c});
                    ds.sort(Comparator.comparingDouble(a -> a[0]));
                    List<Integer> trimmed = new ArrayList<>();
                    for (int i = 0; i < Math.min(maxM, ds.size()); i++)
                        trimmed.add((int) ds.get(i)[1]);
                    nNode.nbrs.set(lc, trimmed);
                }
            }
            if (!W.isEmpty()) ep = (int) W.get(0)[1];
        }
        if (lvl > topLayer) { topLayer = lvl; entryPt = id; }
    }

    public List<float[]> knn(List<Float> q, int k, int ef,
                              BiFunction<List<Float>, List<Float>, Float> dist) {
        if (entryPt == -1) return Collections.emptyList();
        int ep = entryPt;
        for (int lc = topLayer; lc > 0; lc--) {
            Node epNode = G.get(ep);
            if (epNode != null && lc < epNode.nbrs.size()) {
                List<float[]> W = searchLayer(q, ep, 1, lc, dist);
                if (!W.isEmpty()) ep = (int) W.get(0)[1];
            }
        }
        List<float[]> W = searchLayer(q, ep, Math.max(ef, k), 0, dist);
        return W.size() > k ? W.subList(0, k) : W;
    }

    public void remove(int id) {
        if (!G.containsKey(id)) return;
        for (Node nd : G.values())
            for (List<Integer> layer : nd.nbrs)
                layer.removeIf(nid -> nid == id);
        if (entryPt == id) {
            entryPt = -1;
            for (int nid : G.keySet())
                if (nid != id) { entryPt = nid; break; }
        }
        G.remove(id);
    }

    public GraphInfo getInfo() {
        GraphInfo gi = new GraphInfo();
        gi.topLayer = topLayer;
        gi.nodeCount = G.size();
        int maxL = Math.max(topLayer + 1, 1);
        for (int i = 0; i < maxL; i++) { gi.nodesPerLayer.add(0); gi.edgesPerLayer.add(0); }
        for (Map.Entry<Integer, Node> e : G.entrySet()) {
            int id = e.getKey(); Node nd = e.getValue();
            GraphInfo.NodeView nv = new GraphInfo.NodeView();
            nv.id = id; nv.metadata = nd.item.metadata;
            nv.category = nd.item.category; nv.maxLyr = nd.maxLyr;
            gi.nodes.add(nv);
            for (int lc = 0; lc <= nd.maxLyr && lc < maxL; lc++) {
                gi.nodesPerLayer.set(lc, gi.nodesPerLayer.get(lc) + 1);
                if (lc < nd.nbrs.size()) {
                    for (int nid : nd.nbrs.get(lc)) {
                        if (id < nid) {
                            gi.edgesPerLayer.set(lc, gi.edgesPerLayer.get(lc) + 1);
                            GraphInfo.EdgeView ev = new GraphInfo.EdgeView();
                            ev.src = id; ev.dst = nid; ev.lyr = lc;
                            gi.edges.add(ev);
                        }
                    }
                }
            }
        }
        return gi;
    }

    public int size() { return G.size(); }
}
