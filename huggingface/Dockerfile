# Hugging Face Space image: Ollama + the VectorDB Java server in one container.
# The app files come from the GitHub repo's main branch, so a "Factory rebuild"
# of the Space picks up the latest version.

FROM eclipse-temurin:21-jre AS jre

FROM ollama/ollama:0.5.7

COPY --from=jre /opt/java/openjdk /opt/java/openjdk

# Spaces run the container as a non-root user (uid 1000), so models and HOME
# must be world-writable. llama3.2:1b keeps answers quick on the free CPU tier.
ENV JAVA_HOME=/opt/java/openjdk \
    PATH=/opt/java/openjdk/bin:$PATH \
    HOME=/home/user \
    OLLAMA_MODELS=/models \
    OLLAMA_MODEL=llama3.2:1b \
    PORT=7860

# Download both models at build time so the Space starts ready to answer
RUN mkdir -p /models /home/user \
 && (ollama serve >/tmp/ollama-build.log 2>&1 &) \
 && for i in $(seq 1 30); do ollama list >/dev/null 2>&1 && break; sleep 1; done \
 && ollama pull nomic-embed-text \
 && ollama pull "$OLLAMA_MODEL" \
 && chmod -R a+rwX /models /home/user

WORKDIR /app
ADD https://raw.githubusercontent.com/Har-ry123/Own-Ai-Assistant/main/vectordb.jar /app/vectordb.jar
ADD https://raw.githubusercontent.com/Har-ry123/Own-Ai-Assistant/main/index.html /app/index.html
RUN chmod a+r /app/vectordb.jar /app/index.html

EXPOSE 7860

# Start Ollama, wait until it answers, then run the server in the foreground
ENTRYPOINT ["/bin/sh", "-c", "ollama serve >/tmp/ollama.log 2>&1 & for i in $(seq 1 60); do ollama list >/dev/null 2>&1 && break; sleep 1; done; exec java -jar /app/vectordb.jar"]
