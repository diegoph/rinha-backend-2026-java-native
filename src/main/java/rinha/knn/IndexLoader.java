package rinha.knn;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public final class IndexLoader {
    private IndexLoader() {}

    public static FraudIndex load(Path path) throws IOException {
        String s = path.toString();
        if (s.endsWith(".bin")) return IvfIndex.load(path);
        return ReferenceIndex.load(path);
    }

    public static Path defaultPath() {
        String explicit = System.getenv("INDEX_PATH");
        if (explicit != null && !explicit.isBlank()) return Path.of(explicit);

        Path bin = Path.of("/app/resources/index.bin");
        if (Files.exists(bin)) return bin;
        Path gz = Path.of("/app/resources/references.json.gz");
        if (Files.exists(gz)) return gz;
        Path example = Path.of("/app/resources/example-references.json");
        if (Files.exists(example)) return example;

        Path localBin = Path.of("resources/index.bin");
        if (Files.exists(localBin)) return localBin;
        Path localGz = Path.of("resources/references.json.gz");
        if (Files.exists(localGz)) return localGz;
        return Path.of("resources/example-references.json");
    }
}
