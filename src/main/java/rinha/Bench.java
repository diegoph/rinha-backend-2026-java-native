package rinha;

import rinha.knn.FraudIndex;
import rinha.knn.IndexLoader;
import rinha.knn.IvfIndex;
import rinha.vector.Vectorizer;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

public final class Bench {
    private static final byte[] SAMPLE = ("{\"id\":\"tx-1329056812\","
            + "\"transaction\":{\"amount\":41.12,\"installments\":2,\"requested_at\":\"2026-03-11T18:45:53Z\"},"
            + "\"customer\":{\"avg_amount\":82.24,\"tx_count_24h\":3,\"known_merchants\":[\"MERC-003\",\"MERC-016\"]},"
            + "\"merchant\":{\"id\":\"MERC-016\",\"mcc\":\"5411\",\"avg_amount\":60.25},"
            + "\"terminal\":{\"is_online\":false,\"card_present\":true,\"km_from_home\":29.23},"
            + "\"last_transaction\":null}").getBytes(StandardCharsets.US_ASCII);

    public static void main(String[] args) throws Exception {
        Path refs = args.length > 0 ? Path.of(args[0]) : defaultRefs();
        int iterations = Integer.getInteger("iterations", 20_000);
        int warmup = Integer.getInteger("warmup", 2_000);

        FraudIndex index = IndexLoader.load(refs);
        Vectorizer vectorizer = new Vectorizer();
        short[] q = new short[14];

        for (int i = 0; i < warmup; i++) {
            vectorizer.vectorizeToInt16(SAMPLE, SAMPLE.length, q);
            index.searchFraudCount(q);
        }

        long[] totalNs = new long[iterations];
        long[] vectorNs = new long[iterations];
        long[] knnNs = new long[iterations];
        long[] scannedVectors = index instanceof IvfIndex ? new long[iterations] : null;
        long[] scannedClusters = index instanceof IvfIndex ? new long[iterations] : null;
        long[] repairVectors = index instanceof IvfIndex ? new long[iterations] : null;
        long[] bboxChecks = index instanceof IvfIndex ? new long[iterations] : null;
        int frauds = 0;

        for (int i = 0; i < iterations; i++) {
            long t0 = System.nanoTime();
            vectorizer.vectorizeToInt16(SAMPLE, SAMPLE.length, q);
            long t1 = System.nanoTime();
            frauds = index.searchFraudCount(q);
            long t2 = System.nanoTime();
            vectorNs[i] = t1 - t0;
            knnNs[i] = t2 - t1;
            totalNs[i] = t2 - t0;
            if (index instanceof IvfIndex ivf) {
                IvfIndex.SearchStats st = ivf.lastStats();
                scannedVectors[i] = st.scannedVectors;
                scannedClusters[i] = st.scannedClusters;
                repairVectors[i] = st.repairVectors;
                bboxChecks[i] = st.bboxChecks;
            }
        }

        Arrays.sort(vectorNs);
        Arrays.sort(knnNs);
        Arrays.sort(totalNs);
        System.out.printf("refs=%s n=%d frauds=%d iterations=%d%n", refs, index.size(), frauds, iterations);
        print("vector", vectorNs);
        print("knn", knnNs);
        print("total", totalNs);
        if (scannedVectors != null) {
            Arrays.sort(scannedVectors);
            Arrays.sort(scannedClusters);
            Arrays.sort(repairVectors);
            Arrays.sort(bboxChecks);
            printCount("scanVec", scannedVectors);
            printCount("scanCls", scannedClusters);
            printCount("repairV", repairVectors);
            printCount("bboxChk", bboxChecks);
        }
    }

    private static Path defaultRefs() {
        Path bin = Path.of("resources/index.bin");
        if (Files.exists(bin)) return bin;
        Path gz = Path.of("resources/references.json.gz");
        if (Files.exists(gz)) return gz;
        return Path.of("resources/example-references.json");
    }

    private static void print(String name, long[] ns) {
        long sum = 0;
        for (long n : ns) sum += n;
        long avg = sum / ns.length;
        long p50 = ns[(int) (ns.length * 0.50)];
        long p95 = ns[(int) (ns.length * 0.95)];
        long p99 = ns[Math.min(ns.length - 1, (int) (ns.length * 0.99))];
        System.out.printf("%-6s avg=%7.2f us p50=%7.2f us p95=%7.2f us p99=%7.2f us%n",
                name, avg / 1000.0, p50 / 1000.0, p95 / 1000.0, p99 / 1000.0);
    }

    private static void printCount(String name, long[] xs) {
        long sum = 0;
        for (long x : xs) sum += x;
        long avg = sum / xs.length;
        long p50 = xs[(int) (xs.length * 0.50)];
        long p95 = xs[(int) (xs.length * 0.95)];
        long p99 = xs[Math.min(xs.length - 1, (int) (xs.length * 0.99))];
        System.out.printf("%-7s avg=%8d p50=%8d p95=%8d p99=%8d%n", name, avg, p50, p95, p99);
    }
}
