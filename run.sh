#!/bin/bash
# VectorDB Java Server — run script (macOS / Linux / Git Bash)
# Requires: Java 21+. Optional argument: port (default 9999, or $PORT)
cd "$(dirname "$0")"

echo "=== VectorDB Java Server ==="

if ! command -v java >/dev/null 2>&1; then
    echo "Java 21 or newer is required but was not found. Install it from https://adoptium.net" >&2
    exit 1
fi
major=$(java -version 2>&1 | awk -F'"' '/version/ {print $2}' | cut -d. -f1)
if [ "${major:-0}" -lt 21 ]; then
    echo "Java 21 or newer is required (found Java $major). Install it from https://adoptium.net" >&2
    exit 1
fi

# Rebuild the jar from source if it is missing
if [ ! -f "vectordb.jar" ]; then
    echo "Compiling from source ..."
    rm -rf out && mkdir -p out
    javac -d out *.java || { echo "Compilation failed. Fix Java errors and retry." >&2; exit 1; }
    jar cfe vectordb.jar com.vectordb.VectorDBServer -C out .
fi

exec java -jar vectordb.jar "$@"
