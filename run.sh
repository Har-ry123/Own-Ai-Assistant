#!/bin/bash
# VectorDB Java Server — run script
# Requires: Java 21+

echo "=== VectorDB Java Server ==="

# Option 1: Run prebuilt JAR (fastest)
if [ -f "vectordb.jar" ]; then
    echo "Starting from vectordb.jar ..."
    java -jar vectordb.jar
    exit 0
fi

# Option 2: Compile from source then run
echo "Compiling from source ..."
mkdir -p out
javac -d out *.java
if [ $? -ne 0 ]; then
    echo "Compilation failed. Fix Java errors and retry." >&2
    exit 1
fi
echo "Packaging vectordb.jar ..."
jar cfe vectordb.jar com.vectordb.VectorDBServer -C out .
echo "Starting from rebuilt vectordb.jar ..."
java -jar vectordb.jar
exit 0
