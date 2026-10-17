# Own AI Assistant: VectorDB + HNSW + RAG

A vector database written from scratch in Java, with a web UI. It shows how
**HNSW**, **KD-Tree** and **Brute Force** nearest-neighbour search compare, and
uses a local AI model ([Ollama](https://ollama.com)) to answer questions about
documents you add (RAG).

- **Search tab**: search the demo vectors with any of the 3 algorithms and 3
  distance metrics, compare their speed, and see the HNSW graph layers
- **Documents tab**: paste notes or text; they're split into chunks and embedded
  with `nomic-embed-text`
- **Ask AI tab**: ask questions; the most relevant chunks are retrieved and
  `llama3.2` answers using them
- **2D plot**: every vector projected with PCA, with the query and its matches
  highlighted

## Requirements

| What | Needed for | Get it |
|---|---|---|
| **Java 21 or newer** | Everything | [adoptium.net](https://adoptium.net) (pick the JDK, version 21 or later) |
| **Ollama** + 2 models | Documents and Ask AI only (search works without it) | [ollama.com](https://ollama.com) |

Check your Java version with `java -version`. If it says 17 or lower, install
Java 21+. On Windows, `run.bat` finds a newer Java automatically even when an
older one comes first on your PATH.

## Run it

1. **Download** this repository: **Code → Download ZIP** and unzip it, or:

   ```bash
   git clone https://github.com/Har-ry123/Own-Ai-Assistant.git
   cd Own-Ai-Assistant
   ```

2. **Start the server**:

   - **Windows:** double-click `run.bat`, or in a terminal in this folder run:
     ```powershell
     .\run.bat
     ```
   - **macOS / Linux:**
     ```bash
     bash run.sh
     ```

3. **Open** <http://localhost:9999> in your browser.

To use a different port, pass it as an argument (`run.bat 8080` or
`bash run.sh 8080`) or set the `PORT` environment variable. If the port is
busy, the server tries the next 9 ports and prints the one it used.

## Enable the AI features (optional)

1. Install Ollama from [ollama.com](https://ollama.com) and start it.
2. Download the two models (about 2.3 GB in total):

   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3.2
   ```

3. Restart the server. The **OLLAMA** badge in the top bar turns green.

The server expects Ollama at `127.0.0.1:11434`, which is Ollama's default.
To use a different model, set `OLLAMA_MODEL` (answers) or `OLLAMA_EMBED_MODEL`
(embeddings) before starting, for example `OLLAMA_MODEL=llama3.2:1b` for faster
answers on a slow machine.

## Host it online for free (always on)

The [`huggingface/`](huggingface/) folder holds a ready-made
[Hugging Face Space](https://huggingface.co/spaces) setup that runs the server
and Ollama together on Hugging Face's free tier. You get a permanent public link
that works when your own computer is off.

1. Sign in at [huggingface.co](https://huggingface.co) and click **New Space**.
2. Name it (for example `hariom-ai-assistant`), choose **Docker** → **Blank**,
   keep the free **CPU basic** hardware, and create it.
3. In the Space, open **Files** → **Add file** → **Upload files**, and upload
   `huggingface/Dockerfile` and `huggingface/README.md` from this repo. Replace
   the existing README when asked.
4. Wait for the build to finish (about 5–10 minutes, because it downloads the AI
   models). Your app is then live at
   `https://<your-username>-hariom-ai-assistant.hf.space`.

The Space downloads `vectordb.jar` and `index.html` from this repo's `main`
branch. After you change them on GitHub, open the Space's **Settings** and
click **Factory rebuild** to update the live site. Free Spaces go to sleep after
about 48 hours without visitors and wake up on the next visit, which takes about
a minute.

## Share it with a public link (optional)

The app runs on your computer, so anyone you share with uses your Java server
and your Ollama. [Cloudflare quick tunnels](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/)
give you a free temporary link without an account:

```bash
cloudflared tunnel --url http://localhost:9999
```

It prints a `https://….trycloudflare.com` link that works while both programs
keep running. There's no login, so anyone with the link can add and delete
data. Everything is kept in memory and is lost when the server stops.

## Project layout

| File | What it does |
|---|---|
| `VectorDBServer.java` | HTTP server and API endpoints; loads the 20 demo vectors |
| `VectorDB.java` | Demo vector store; keeps HNSW, KD-Tree and Brute Force indexes in sync |
| `HNSW.java` | Hierarchical Navigable Small World graph index |
| `KDTree.java` | KD-Tree index (exact only for Euclidean distance) |
| `BruteForce.java` | Exact linear scan |
| `DistanceMetrics.java` | Euclidean, cosine and Manhattan distance |
| `DocumentDB.java` | Stores document chunks with their real 768-dimension embeddings |
| `TextChunker.java` | Splits text into overlapping 250-word chunks |
| `OllamaClient.java` | Calls Ollama for embeddings and answers |
| `VectorItem.java` | A stored vector with its metadata |
| `index.html` | The whole web UI (served by the Java server) |
| `vectordb.jar` | Prebuilt server. Delete it and the run scripts rebuild it from source |

### Rebuild the jar yourself

```bash
javac -d out *.java
jar cfe vectordb.jar com.vectordb.VectorDBServer -C out .
```

## API

All endpoints return JSON.

| Method | Path | Purpose |
|---|---|---|
| GET | `/status` | Ollama status, model names, counts |
| GET | `/items` | All demo vectors |
| GET | `/search?v=…&k=5&metric=cosine&algo=hnsw` | k nearest neighbours of a 16-number vector (`algo`: `hnsw`, `kdtree`, `bruteforce`; `metric`: `cosine`, `euclidean`, `manhattan`) |
| GET | `/benchmark?v=…&k=5&metric=cosine` | Time all 3 algorithms on the same query |
| GET | `/hnsw-info` | HNSW layers, nodes and edges |
| POST | `/insert` | Add a demo vector: `{"metadata","category","embedding":[16 numbers]}` |
| DELETE | `/delete/{id}` | Remove a demo vector |
| POST | `/doc/insert` | Add a document: `{"title","text"}` |
| GET | `/doc/list` | List stored document chunks |
| DELETE | `/doc/delete/{id}` | Remove a document chunk |
| POST | `/doc/search` | Find the chunks closest to `{"question","k"}` |
| POST | `/doc/ask` | Answer `{"question","k"}` using the closest chunks |
