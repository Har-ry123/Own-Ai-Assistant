package com.vectordb;

import com.sun.net.httpserver.*;
import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.BiFunction;
import java.util.stream.*;

public class VectorDBServer {

    private static final int DIMS = 16;
    private static final VectorDB db = new VectorDB(DIMS);
    private static final DocumentDB docDB = new DocumentDB();
    private static final OllamaClient ollama = new OllamaClient("127.0.0.1", 11434);

    public static void main(String[] args) throws Exception {
        loadDemo(db);
        boolean ollamaUp = ollama.isAvailable();
        System.out.println("=== VectorDB Engine (Java) ===");

        // Determine base port: command-line arg > PORT env var > default 8080
        int basePort = 8080;
        if (args.length > 0) {
            try {
                basePort = Integer.parseInt(args[0]);
            } catch (NumberFormatException ignored) {
            }
        } else {
            String p = System.getenv("PORT");
            if (p != null) {
                try {
                    basePort = Integer.parseInt(p);
                } catch (NumberFormatException ignored) {
                }
            }
        }

        System.out.println("Attempting to bind HTTP server on starting port " + basePort + " (will try nearby ports if in use)");
        System.out.println(db.size() + " demo vectors | " + DIMS + " dims | HNSW+KD-Tree+BruteForce");
        System.out.println("Ollama: " + (ollamaUp ? "ONLINE" : "OFFLINE (install from ollama.com)"));

        HttpServer server = null;
        int boundPort = -1;
        for (int port = basePort; port < basePort + 10; port++) {
            try {
                server = HttpServer.create(new InetSocketAddress("0.0.0.0", port), 0);
                boundPort = port;
                break;
            } catch (java.io.IOException e) {
                if (e instanceof java.net.BindException) {
                    System.err.println("Port " + port + " in use, trying next port...");
                    continue;
                } else {
                    throw e;
                }
            }
        }
        if (server == null) {
            throw new RuntimeException("Failed to bind HTTP server on ports " + basePort + "-" + (basePort + 9));
        }
        System.out.println("http://localhost:" + boundPort);
        server.createContext("/", new RootHandler());
        server.createContext("/search", new SearchHandler());
        server.createContext("/insert", new InsertHandler());
        server.createContext("/delete/", new DeleteHandler());
        server.createContext("/items", new ItemsHandler());
        server.createContext("/benchmark", new BenchmarkHandler());
        server.createContext("/hnsw-info", new HnswInfoHandler());
        server.createContext("/stats", new StatsHandler());
        server.createContext("/status", new StatusHandler());
        server.createContext("/doc/insert", new DocInsertHandler());
        server.createContext("/doc/delete/", new DocDeleteHandler());
        server.createContext("/doc/list", new DocListHandler());
        server.createContext("/doc/search", new DocSearchHandler());
        server.createContext("/doc/ask", new DocAskHandler());

        server.setExecutor(Executors.newFixedThreadPool(8));
        server.start();
        System.out.println("Server started on port " + boundPort + ". Press Ctrl+C to stop.");
        // Keep main thread alive
        Thread.currentThread().join();
    }

    // ── UTILITIES ────────────────────────────────────────────────────
    static void cors(HttpExchange ex) throws IOException {
        ex.getResponseHeaders().set("Access-Control-Allow-Origin", "*");
        ex.getResponseHeaders().set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        ex.getResponseHeaders().set("Access-Control-Allow-Headers", "Content-Type");
        ex.getResponseHeaders().set("Content-Type", "application/json");
    }

    static void send(HttpExchange ex, int code, String json) throws IOException {
        byte[] b = json.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().set("Content-Type", "application/json");
        ex.sendResponseHeaders(code, b.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(b);
        }
    }

    static String readBody(HttpExchange ex) throws IOException {
        return new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
    }

    static Map<String, String> queryParams(HttpExchange ex) {
        Map<String, String> params = new HashMap<>();
        String query = ex.getRequestURI().getQuery();
        if (query == null) {
            return params;
        }
        for (String p : query.split("&")) {
            String[] kv = p.split("=", 2);
            if (kv.length == 2) {
                try {
                    params.put(kv[0], URLDecoder.decode(kv[1], StandardCharsets.UTF_8));
                } catch (Exception e) {
                    params.put(kv[0], kv[1]);
                }
            }
        }
        return params;
    }

    static List<Float> parseVec(String s) {
        if (s == null || s.isEmpty()) {
            return List.of();
        }
        List<Float> r = new ArrayList<>();
        for (String t : s.split(",")) {
            try {
                r.add(Float.parseFloat(t.trim()));
            } catch (NumberFormatException e) {
            }
        }
        return r;
    }

    static String jS(String s) {
        if (s == null) {
            return "\"\"";
        }
        return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"")
                .replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t") + "\"";
    }

    static String jVec(List<Float> v) {
        return "[" + v.stream().map(f -> String.format("%.4f", f)).collect(Collectors.joining(",")) + "]";
    }

    // Simple JSON body parsers
    static String extractStr(String body, String key) {
        int p = body.indexOf("\"" + key + "\"");
        if (p < 0) {
            return "";
        }
        p = body.indexOf(':', p) + 1;
        while (p < body.length() && (body.charAt(p) == ' ' || body.charAt(p) == '\t')) {
            p++;
        }
        if (p >= body.length() || body.charAt(p) != '"') {
            return "";
        }
        p++;
        StringBuilder sb = new StringBuilder();
        while (p < body.length()) {
            char c = body.charAt(p);
            if (c == '"') {
                break;
            }
            if (c == '\\' && p + 1 < body.length()) {
                p++;
                switch (body.charAt(p)) {
                    case '"' ->
                        sb.append('"');
                    case '\\' ->
                        sb.append('\\');
                    case 'n' ->
                        sb.append('\n');
                    case 'r' ->
                        sb.append('\r');
                    case 't' ->
                        sb.append('\t');
                    default ->
                        sb.append(body.charAt(p));
                }
            } else {
                sb.append(c);
            }
            p++;
        }
        return sb.toString();
    }

    static int extractInt(String body, String key, int def) {
        int p = body.indexOf("\"" + key + "\"");
        if (p < 0) {
            return def;
        }
        p = body.indexOf(':', p) + 1;
        while (p < body.length() && (body.charAt(p) == ' ' || body.charAt(p) == '\t')) {
            p++;
        }
        try {
            return Integer.parseInt(body.substring(p).replaceAll("[^0-9].*", "").trim());
        } catch (NumberFormatException e) {
            return def;
        }
    }

    static List<Float> extractArr(String body, String key) {
        int p = body.indexOf("\"" + key + "\"");
        if (p < 0) {
            return List.of();
        }
        p = body.indexOf('[', p);
        if (p < 0) {
            return List.of();
        }
        int e = body.indexOf(']', p);
        if (e < 0) {
            return List.of();
        }
        return parseVec(body.substring(p + 1, e));
    }

    // ── HANDLERS ─────────────────────────────────────────────────────
    static class RootHandler implements HttpHandler {

        @Override
        public void handle(HttpExchange ex) throws IOException {
            if ("OPTIONS".equals(ex.getRequestMethod())) {
                cors(ex);
                ex.sendResponseHeaders(204, -1);
                return;
            }
            // Serve index.html from same dir as JAR, or classpath
            String html;
            File f = new File("index.html");
            if (f.exists()) {
                html = Files.readString(f.toPath());
            } else {
                html = "<h1>index.html not found</h1>";
            }
            byte[] b = html.getBytes(StandardCharsets.UTF_8);
            ex.getResponseHeaders().set("Content-Type", "text/html; charset=utf-8");
            ex.sendResponseHeaders(200, b.length);
            try (OutputStream os = ex.getResponseBody()) {
                os.write(b);
            }
        }
    }

    static class SearchHandler implements HttpHandler {

        @Override
        public void handle(HttpExchange ex) throws IOException {
            cors(ex);
            if ("OPTIONS".equals(ex.getRequestMethod())) {
                ex.sendResponseHeaders(204, -1);
                return;
            }
            var p = queryParams(ex);
            var q = parseVec(p.get("v"));
            if (q.size() != DIMS) {
                send(ex, 400, "{\"error\":\"need " + DIMS + "D vector\"}");
                return;
            }
            int k = 5;
            try {
                k = Integer.parseInt(p.getOrDefault("k", "5"));
            } catch (NumberFormatException ignored) {
            }
            String metric = p.getOrDefault("metric", "cosine");
            String algo = p.getOrDefault("algo", "hnsw");
            var out = db.search(q, k, metric, algo);
            StringBuilder sb = new StringBuilder("{\"results\":[");
            for (int i = 0; i < out.hits().size(); i++) {
                if (i > 0) {
                    sb.append(',');
                }
                var h = out.hits().get(i);
                sb.append("{\"id\":").append(h.id()).append(",\"metadata\":").append(jS(h.meta()))
                        .append(",\"category\":").append(jS(h.cat()))
                        .append(",\"distance\":").append(String.format("%.6f", h.dist()))
                        .append(",\"embedding\":").append(jVec(h.emb())).append('}');
            }
            sb.append("],\"latencyUs\":").append(out.us()).append(",\"algo\":").append(jS(out.algo()))
                    .append(",\"metric\":").append(jS(out.metric())).append('}');
            send(ex, 200, sb.toString());
        }
    }

    static class InsertHandler implements HttpHandler {

        @Override
        public void handle(HttpExchange ex) throws IOException {
            cors(ex);
            if ("OPTIONS".equals(ex.getRequestMethod())) {
                ex.sendResponseHeaders(204, -1);
                return;
            }
            String body = readBody(ex);
            String meta = extractStr(body, "metadata"), cat = extractStr(body, "category");
            List<Float> emb = extractArr(body, "embedding");
            if (meta.isEmpty() || emb.size() != DIMS) {
                send(ex, 400, "{\"error\":\"invalid body\"}");
                return;
            }
            int id = db.insert(meta, cat, emb, DistanceMetrics::cosine);
            send(ex, 200, "{\"id\":" + id + "}");
        }
    }

    static class DeleteHandler implements HttpHandler {

        @Override
        public void handle(HttpExchange ex) throws IOException {
            cors(ex);
            if ("OPTIONS".equals(ex.getRequestMethod())) {
                ex.sendResponseHeaders(204, -1);
                return;
            }
            String path = ex.getRequestURI().getPath();
            try {
                int id = Integer.parseInt(path.replaceAll(".*/", ""));
                send(ex, 200, "{\"ok\":" + db.remove(id) + "}");
            } catch (NumberFormatException e) {
                send(ex, 400, "{\"error\":\"bad id\"}");
            }
        }
    }

    static class ItemsHandler implements HttpHandler {

        @Override
        public void handle(HttpExchange ex) throws IOException {
            cors(ex);
            var items = db.all();
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < items.size(); i++) {
                if (i > 0) {
                    sb.append(',');
                }
                var v = items.get(i);
                sb.append("{\"id\":").append(v.id).append(",\"metadata\":").append(jS(v.metadata))
                        .append(",\"category\":").append(jS(v.category))
                        .append(",\"embedding\":").append(jVec(v.emb)).append('}');
            }
            send(ex, 200, sb.append(']').toString());
        }
    }

    static class BenchmarkHandler implements HttpHandler {

        @Override
        public void handle(HttpExchange ex) throws IOException {
            cors(ex);
            var p = queryParams(ex);
            var q = parseVec(p.get("v"));
            if (q.size() != DIMS) {
                send(ex, 400, "{\"error\":\"need " + DIMS + "D vector\"}");
                return;
            }
            int k = 5;
            try {
                k = Integer.parseInt(p.getOrDefault("k", "5"));
            } catch (NumberFormatException ignored) {
            }
            String metric = p.getOrDefault("metric", "cosine");
            var b = db.benchmark(q, k, metric);
            send(ex, 200, "{\"bruteforceUs\":" + b.bfUs() + ",\"kdtreeUs\":" + b.kdUs()
                    + ",\"hnswUs\":" + b.hnswUs() + ",\"itemCount\":" + b.n() + "}");
        }
    }

    static class HnswInfoHandler implements HttpHandler {

        @Override
        public void handle(HttpExchange ex) throws IOException {
            cors(ex);
            var gi = db.hnswInfo();
            StringBuilder sb = new StringBuilder();
            sb.append("{\"topLayer\":").append(gi.topLayer).append(",\"nodeCount\":").append(gi.nodeCount);
            sb.append(",\"nodesPerLayer\":[").append(gi.nodesPerLayer.stream().map(Object::toString).collect(Collectors.joining(","))).append("]");
            sb.append(",\"edgesPerLayer\":[").append(gi.edgesPerLayer.stream().map(Object::toString).collect(Collectors.joining(","))).append("]");
            sb.append(",\"nodes\":[");
            for (int i = 0; i < gi.nodes.size(); i++) {
                if (i > 0) {
                    sb.append(',');
                }
                var n = gi.nodes.get(i);
                sb.append("{\"id\":").append(n.id).append(",\"metadata\":").append(jS(n.metadata))
                        .append(",\"category\":").append(jS(n.category)).append(",\"maxLyr\":").append(n.maxLyr).append('}');
            }
            sb.append("],\"edges\":[");
            for (int i = 0; i < gi.edges.size(); i++) {
                if (i > 0) {
                    sb.append(',');
                }
                var e = gi.edges.get(i);
                sb.append("{\"src\":").append(e.src).append(",\"dst\":").append(e.dst).append(",\"lyr\":").append(e.lyr).append('}');
            }
            sb.append("]}");
            send(ex, 200, sb.toString());
        }
    }

    static class StatsHandler implements HttpHandler {

        @Override
        public void handle(HttpExchange ex) throws IOException {
            cors(ex);
            send(ex, 200, "{\"count\":" + db.size() + ",\"dims\":" + DIMS
                    + ",\"algorithms\":[\"bruteforce\",\"kdtree\",\"hnsw\"],\"metrics\":[\"euclidean\",\"cosine\",\"manhattan\"]}");
        }
    }

    static class StatusHandler implements HttpHandler {

        @Override
        public void handle(HttpExchange ex) throws IOException {
            cors(ex);
            boolean up = ollama.isAvailable();
            send(ex, 200, "{\"ollamaAvailable\":" + up + ",\"embedModel\":" + jS(ollama.embedModel)
                    + ",\"genModel\":" + jS(ollama.genModel) + ",\"docCount\":" + docDB.size()
                    + ",\"docDims\":" + docDB.getDims() + ",\"demoDims\":" + DIMS + ",\"demoCount\":" + db.size() + "}");
        }
    }

    static class DocInsertHandler implements HttpHandler {

        @Override
        public void handle(HttpExchange ex) throws IOException {
            cors(ex);
            if ("OPTIONS".equals(ex.getRequestMethod())) {
                ex.sendResponseHeaders(204, -1);
                return;
            }
            String body = readBody(ex);
            String title = extractStr(body, "title"), text = extractStr(body, "text");
            if (title.isEmpty() || text.isEmpty()) {
                send(ex, 400, "{\"error\":\"need title and text\"}");
                return;
            }
            List<String> chunks = TextChunker.chunk(text, 250, 30);
            List<Integer> ids = new ArrayList<>();
            for (int i = 0; i < chunks.size(); i++) {
                List<Float> emb = ollama.embed(chunks.get(i));
                if (emb.isEmpty()) {
                    send(ex, 400, "{\"error\":\"Ollama unavailable\"}");
                    return;
                }
                String chunkTitle = chunks.size() > 1 ? title + " [" + (i + 1) + "/" + chunks.size() + "]" : title;
                ids.add(docDB.insert(chunkTitle, chunks.get(i), emb));
            }
            String idsJson = "[" + ids.stream().map(Object::toString).collect(Collectors.joining(",")) + "]";
            send(ex, 200, "{\"ids\":" + idsJson + ",\"chunks\":" + chunks.size() + ",\"dims\":" + docDB.getDims() + "}");
        }
    }

    static class DocDeleteHandler implements HttpHandler {

        @Override
        public void handle(HttpExchange ex) throws IOException {
            cors(ex);
            if ("OPTIONS".equals(ex.getRequestMethod())) {
                ex.sendResponseHeaders(204, -1);
                return;
            }
            String path = ex.getRequestURI().getPath();
            try {
                int id = Integer.parseInt(path.replaceAll(".*/", ""));
                send(ex, 200, "{\"ok\":" + docDB.remove(id) + "}");
            } catch (NumberFormatException e) {
                send(ex, 400, "{\"error\":\"bad id\"}");
            }
        }
    }

    static class DocListHandler implements HttpHandler {

        @Override
        public void handle(HttpExchange ex) throws IOException {
            cors(ex);
            var docs = docDB.all();
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < docs.size(); i++) {
                if (i > 0) {
                    sb.append(',');
                }
                var d = docs.get(i);
                String preview = d.text.length() > 120 ? d.text.substring(0, 120) + "…" : d.text;
                int words = d.text.trim().split("\\s+").length;
                sb.append("{\"id\":").append(d.id).append(",\"title\":").append(jS(d.title))
                        .append(",\"preview\":").append(jS(preview)).append(",\"words\":").append(words).append('}');
            }
            send(ex, 200, sb.append(']').toString());
        }
    }

    static class DocSearchHandler implements HttpHandler {

        @Override
        public void handle(HttpExchange ex) throws IOException {
            cors(ex);
            if ("OPTIONS".equals(ex.getRequestMethod())) {
                ex.sendResponseHeaders(204, -1);
                return;
            }
            String body = readBody(ex);
            String question = extractStr(body, "question");
            int k = extractInt(body, "k", 3);
            if (question.isEmpty()) {
                send(ex, 400, "{\"error\":\"need question\"}");
                return;
            }
            List<Float> qEmb = ollama.embed(question);
            if (qEmb.isEmpty()) {
                send(ex, 400, "{\"error\":\"Ollama unavailable\"}");
                return;
            }
            var hits = docDB.search(qEmb, k, 0.7f);
            StringBuilder sb = new StringBuilder("{\"contexts\":[");
            for (int i = 0; i < hits.size(); i++) {
                if (i > 0) {
                    sb.append(',');
                }
                sb.append("{\"id\":").append(hits.get(i).getValue().id)
                        .append(",\"title\":").append(jS(hits.get(i).getValue().title))
                        .append(",\"distance\":").append(String.format("%.4f", hits.get(i).getKey())).append('}');
            }
            send(ex, 200, sb.append("]}").toString());
        }
    }

    static class DocAskHandler implements HttpHandler {

        @Override
        public void handle(HttpExchange ex) throws IOException {
            cors(ex);
            if ("OPTIONS".equals(ex.getRequestMethod())) {
                ex.sendResponseHeaders(204, -1);
                return;
            }
            String body = readBody(ex);
            String question = extractStr(body, "question");
            int k = extractInt(body, "k", 3);
            if (question.isEmpty()) {
                send(ex, 400, "{\"error\":\"need question\"}");
                return;
            }
            List<Float> qEmb = ollama.embed(question);
            if (qEmb.isEmpty()) {
                send(ex, 400, "{\"error\":\"Ollama unavailable\"}");
                return;
            }
            var hits = docDB.search(qEmb, k, 0.7f);
            StringBuilder ctx = new StringBuilder();
            for (int i = 0; i < hits.size(); i++) {
                ctx.append("[").append(i + 1).append("] ").append(hits.get(i).getValue().title)
                        .append(":\n").append(hits.get(i).getValue().text).append("\n\n");
            }
            String prompt = "You are a helpful assistant. Answer the user's question directly. "
                    + "Use the provided context if it contains relevant information. "
                    + "If it doesn't, just use your own general knowledge. "
                    + "IMPORTANT: Do NOT mention the 'context'. Just answer naturally.\n\nContext:\n"
                    + ctx + "Question: " + question + "\n\nAnswer:";
            String answer = ollama.generate(prompt);
            StringBuilder sb = new StringBuilder("{\"answer\":").append(jS(answer))
                    .append(",\"model\":").append(jS(ollama.genModel)).append(",\"contexts\":[");
            for (int i = 0; i < hits.size(); i++) {
                if (i > 0) {
                    sb.append(',');
                }
                var h = hits.get(i);
                sb.append("{\"id\":").append(h.getValue().id).append(",\"title\":").append(jS(h.getValue().title))
                        .append(",\"text\":").append(jS(h.getValue().text))
                        .append(",\"distance\":").append(String.format("%.4f", h.getKey())).append('}');
            }
            sb.append("],\"docCount\":").append(docDB.size()).append('}');
            send(ex, 200, sb.toString());
        }
    }

    // ── DEMO DATA ─────────────────────────────────────────────────────
    private static void loadDemo(VectorDB db) {
        var dist = DistanceMetrics.getDistFn("cosine");
        db.insert("Linked List: nodes connected by pointers", "cs", List.of(0.90f, 0.85f, 0.72f, 0.68f, 0.12f, 0.08f, 0.15f, 0.10f, 0.05f, 0.08f, 0.06f, 0.09f, 0.07f, 0.11f, 0.08f, 0.06f), dist);
        db.insert("Binary Search Tree: O(log n) search and insert", "cs", List.of(0.88f, 0.82f, 0.78f, 0.74f, 0.15f, 0.10f, 0.08f, 0.12f, 0.06f, 0.07f, 0.08f, 0.05f, 0.09f, 0.06f, 0.07f, 0.10f), dist);
        db.insert("Dynamic Programming: memoization overlapping subproblems", "cs", List.of(0.82f, 0.76f, 0.88f, 0.80f, 0.20f, 0.18f, 0.12f, 0.09f, 0.07f, 0.06f, 0.08f, 0.07f, 0.08f, 0.09f, 0.06f, 0.07f), dist);
        db.insert("Graph BFS and DFS: breadth and depth first traversal", "cs", List.of(0.85f, 0.80f, 0.75f, 0.82f, 0.18f, 0.14f, 0.10f, 0.08f, 0.06f, 0.09f, 0.07f, 0.06f, 0.10f, 0.08f, 0.09f, 0.07f), dist);
        db.insert("Hash Table: O(1) lookup with collision chaining", "cs", List.of(0.87f, 0.78f, 0.70f, 0.76f, 0.13f, 0.11f, 0.09f, 0.14f, 0.08f, 0.07f, 0.06f, 0.08f, 0.07f, 0.10f, 0.08f, 0.09f), dist);
        db.insert("Calculus: derivatives integrals and limits", "math", List.of(0.12f, 0.15f, 0.18f, 0.10f, 0.91f, 0.86f, 0.78f, 0.72f, 0.08f, 0.06f, 0.07f, 0.09f, 0.07f, 0.08f, 0.06f, 0.10f), dist);
        db.insert("Linear Algebra: matrices eigenvalues eigenvectors", "math", List.of(0.20f, 0.18f, 0.15f, 0.12f, 0.88f, 0.90f, 0.82f, 0.76f, 0.09f, 0.07f, 0.08f, 0.06f, 0.10f, 0.07f, 0.08f, 0.09f), dist);
        db.insert("Probability: distributions random variables Bayes theorem", "math", List.of(0.15f, 0.12f, 0.20f, 0.18f, 0.84f, 0.80f, 0.88f, 0.82f, 0.07f, 0.08f, 0.06f, 0.10f, 0.09f, 0.06f, 0.09f, 0.08f), dist);
        db.insert("Number Theory: primes modular arithmetic RSA cryptography", "math", List.of(0.22f, 0.16f, 0.14f, 0.20f, 0.80f, 0.85f, 0.76f, 0.90f, 0.08f, 0.09f, 0.07f, 0.06f, 0.08f, 0.10f, 0.07f, 0.06f), dist);
        db.insert("Combinatorics: permutations combinations generating functions", "math", List.of(0.18f, 0.20f, 0.16f, 0.14f, 0.86f, 0.78f, 0.84f, 0.80f, 0.06f, 0.07f, 0.09f, 0.08f, 0.06f, 0.09f, 0.10f, 0.07f), dist);
        db.insert("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food", List.of(0.08f, 0.06f, 0.09f, 0.07f, 0.07f, 0.08f, 0.06f, 0.09f, 0.90f, 0.86f, 0.78f, 0.72f, 0.08f, 0.06f, 0.09f, 0.07f), dist);
        db.insert("Sushi: vinegared rice raw fish and nori rolls", "food", List.of(0.06f, 0.08f, 0.07f, 0.09f, 0.09f, 0.06f, 0.08f, 0.07f, 0.86f, 0.90f, 0.82f, 0.76f, 0.07f, 0.09f, 0.06f, 0.08f), dist);
        db.insert("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food", List.of(0.09f, 0.07f, 0.06f, 0.08f, 0.08f, 0.09f, 0.07f, 0.06f, 0.82f, 0.78f, 0.90f, 0.84f, 0.09f, 0.07f, 0.08f, 0.06f), dist);
        db.insert("Tacos: corn tortillas with carnitas salsa and cilantro", "food", List.of(0.07f, 0.09f, 0.08f, 0.06f, 0.06f, 0.07f, 0.09f, 0.08f, 0.78f, 0.82f, 0.86f, 0.90f, 0.06f, 0.08f, 0.07f, 0.09f), dist);
        db.insert("Croissant: laminated pastry with buttery flaky layers", "food", List.of(0.06f, 0.07f, 0.10f, 0.09f, 0.10f, 0.06f, 0.07f, 0.10f, 0.85f, 0.80f, 0.76f, 0.82f, 0.09f, 0.07f, 0.10f, 0.06f), dist);
        db.insert("Basketball: fast-paced shooting dribbling slam dunks", "sports", List.of(0.09f, 0.07f, 0.08f, 0.10f, 0.08f, 0.09f, 0.07f, 0.06f, 0.08f, 0.07f, 0.09f, 0.06f, 0.91f, 0.85f, 0.78f, 0.72f), dist);
        db.insert("Football: tackles touchdowns field goals and strategy", "sports", List.of(0.07f, 0.09f, 0.06f, 0.08f, 0.09f, 0.07f, 0.10f, 0.08f, 0.07f, 0.09f, 0.08f, 0.07f, 0.87f, 0.89f, 0.82f, 0.76f), dist);
        db.insert("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports", List.of(0.08f, 0.06f, 0.09f, 0.07f, 0.07f, 0.08f, 0.06f, 0.09f, 0.09f, 0.06f, 0.07f, 0.08f, 0.83f, 0.80f, 0.88f, 0.82f), dist);
        db.insert("Chess: openings endgames tactics strategic board game", "sports", List.of(0.25f, 0.20f, 0.22f, 0.18f, 0.22f, 0.18f, 0.20f, 0.15f, 0.06f, 0.08f, 0.07f, 0.09f, 0.80f, 0.84f, 0.78f, 0.90f), dist);
        db.insert("Swimming: butterfly freestyle backstroke Olympic competition", "sports", List.of(0.06f, 0.08f, 0.07f, 0.09f, 0.08f, 0.06f, 0.09f, 0.07f, 0.10f, 0.08f, 0.06f, 0.07f, 0.85f, 0.82f, 0.86f, 0.80f), dist);
    }
}
