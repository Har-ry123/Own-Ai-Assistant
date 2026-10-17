package com.vectordb;

import java.net.URI;
import java.net.http.*;
import java.net.http.HttpRequest.BodyPublishers;
import java.net.http.HttpResponse.BodyHandlers;
import java.time.Duration;
import java.util.*;

// Minimal client for a local Ollama server (https://ollama.com)
public class OllamaClient {
    private static final String UNAVAILABLE = "ERROR: Ollama unavailable. Run: ollama serve";

    private final String host;
    private final int port;
    public String embedModel = "nomic-embed-text";
    public String genModel = "llama3.2";

    public OllamaClient(String host, int port) {
        this.host = host;
        this.port = port;
    }

    private String url(String path) {
        return "http://" + host + ":" + port + path;
    }

    private HttpClient newClient(int connectTimeoutSec) {
        return HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(connectTimeoutSec)).build();
    }

    private String esc(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"")
                .replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t");
    }

    public boolean isAvailable() {
        try {
            HttpRequest req = HttpRequest.newBuilder().uri(URI.create(url("/api/tags"))).GET().build();
            return newClient(2).send(req, BodyHandlers.ofString()).statusCode() == 200;
        } catch (Exception e) {
            return false;
        }
    }

    // Returns an empty list if Ollama is unreachable or the model is missing
    public List<Float> embed(String text) {
        try {
            String body = "{\"model\":\"" + esc(embedModel) + "\",\"prompt\":\"" + esc(text) + "\"}";
            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(url("/api/embeddings")))
                    .header("Content-Type", "application/json")
                    .POST(BodyPublishers.ofString(body))
                    .timeout(Duration.ofSeconds(30))
                    .build();
            HttpResponse<String> res = newClient(3).send(req, BodyHandlers.ofString());
            if (res.statusCode() != 200) return List.of();
            return parseEmbedding(res.body());
        } catch (Exception e) {
            return List.of();
        }
    }

    public String generate(String prompt) {
        try {
            String body = "{\"model\":\"" + esc(genModel) + "\",\"prompt\":\"" + esc(prompt) + "\",\"stream\":false}";
            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create(url("/api/generate")))
                    .header("Content-Type", "application/json")
                    .POST(BodyPublishers.ofString(body))
                    .timeout(Duration.ofSeconds(180))
                    .build();
            HttpResponse<String> res = newClient(3).send(req, BodyHandlers.ofString());
            if (res.statusCode() != 200) return UNAVAILABLE;
            return parseResponse(res.body());
        } catch (Exception e) {
            return UNAVAILABLE;
        }
    }

    private List<Float> parseEmbedding(String json) {
        int p = json.indexOf("\"embedding\"");
        if (p < 0) return List.of();
        p = json.indexOf('[', p);
        if (p < 0) return List.of();
        int depth = 1, e = p + 1;
        for (; e < json.length() && depth > 0; e++) {
            if (json.charAt(e) == '[') depth++;
            else if (json.charAt(e) == ']') depth--;
        }
        List<Float> v = new ArrayList<>();
        for (String t : json.substring(p + 1, e - 1).split(",")) {
            try { v.add(Float.parseFloat(t.trim())); }
            catch (NumberFormatException ignored) {}
        }
        return v;
    }

    private String parseResponse(String json) {
        int p = json.indexOf("\"response\"");
        if (p < 0) return "";
        p = json.indexOf(':', p) + 1;
        while (p < json.length() && (json.charAt(p) == ' ' || json.charAt(p) == '\t')) p++;
        if (p >= json.length() || json.charAt(p) != '"') return "";
        p++;
        StringBuilder sb = new StringBuilder();
        for (; p < json.length(); p++) {
            char c = json.charAt(p);
            if (c == '"') break;
            if (c == '\\' && p + 1 < json.length()) {
                p++;
                switch (json.charAt(p)) {
                    case '"'  -> sb.append('"');
                    case '\\' -> sb.append('\\');
                    case 'n'  -> sb.append('\n');
                    case 'r'  -> sb.append('\r');
                    case 't'  -> sb.append('\t');
                    default   -> sb.append(json.charAt(p));
                }
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }
}
