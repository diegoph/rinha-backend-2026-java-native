FROM ghcr.io/graalvm/native-image-community:25 AS build
WORKDIR /src

COPY src ./src
COPY scripts ./scripts
COPY resources/references.json.gz ./resources/references.json.gz

RUN chmod +x scripts/build-index.sh \
    && rm -rf out app.jar /app/rinha-native resources/index.bin \
    && find src/main/java -name '*.java' > sources.txt \
    && javac --release 25 -d out @sources.txt \
    && jar --create --file app.jar --main-class rinha.Main -C out . \
    && ./scripts/build-index.sh resources/references.json.gz resources/index.bin \
    && native-image --no-fallback -O3 -jar app.jar -o /app/rinha-native

FROM debian:bookworm-slim
WORKDIR /app

COPY --from=build /app/rinha-native /app/rinha-native
COPY --from=build /src/resources/index.bin /app/resources/index.bin

ENTRYPOINT ["/app/rinha-native"]
