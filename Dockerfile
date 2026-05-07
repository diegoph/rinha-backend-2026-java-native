FROM ghcr.io/graalvm/native-image-community:25 AS build
WORKDIR /src
COPY src ./src
RUN rm -rf out app.jar /app/rinha-native \
    && find src/main/java -name '*.java' > sources.txt \
    && javac --release 25 -d out @sources.txt \
    && jar --create --file app.jar --main-class rinha.Main -C out . \
    && native-image --no-fallback -O3 -jar app.jar -o /app/rinha-native

FROM debian:bookworm-slim
WORKDIR /app
COPY --from=build /app/rinha-native /app/rinha-native
COPY resources /app/resources
ENTRYPOINT ["/app/rinha-native"]
