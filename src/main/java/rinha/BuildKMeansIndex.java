package rinha;

import rinha.knn.ReferenceIndex;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;


public final class BuildKMeansIndex {
    private static final int DIMS = 14;
    private static final int SCALE = 10_000;
    private static final int MAGIC = 0x4956464A;
    private static final int VERSION = 5;

    private static final int K = Integer.getInteger("kmeans.k", 256);
    private static final int SAMPLE = Integer.getInteger("kmeans.sample", 131_072);
    private static final int ITERS = Integer.getInteger("kmeans.iters", 20);
    private static final long SEED = Long.getLong("kmeans.seed", 8587310L);

    public static void main(String[] args) throws Exception {
        Path refs = args.length > 0 ? Path.of(args[0]) : Path.of("resources/references.json.gz");
        Path out = args.length > 1 ? Path.of(args[1]) : Path.of("resources/index.bin");

        long t0 = System.nanoTime();
        ReferenceIndex src = ReferenceIndex.load(refs);
        int n = src.size();
        short[][] dims = new short[DIMS][];
        for (int d = 0; d < DIMS; d++) dims[d] = src.dim(d);
        byte[] srcLabels = src.labels();
        System.err.printf("loaded refs=%s n=%d loadMs=%d%n", refs, n, ms(t0));
        System.err.printf("kmeans k=%d sample=%d iters=%d seed=%d%n", K, Math.min(SAMPLE, n), ITERS, SEED);

        long t1 = System.nanoTime();
        int[] sampleIdx = makeSample(n, Math.min(SAMPLE, n), SEED);
        short[][] centroids = initCentroids(dims, sampleIdx, K, SEED);
        runLloyd(dims, sampleIdx, centroids, ITERS);
        System.err.printf("trained centroids trainMs=%d%n", ms(t1));

        long t2 = System.nanoTime();
        int[] counts = new int[K];
        for (int i = 0; i < n; i++) counts[nearest(dims, i, centroids)]++;
        int[] offsets = new int[K + 1];
        for (int c = 0; c < K; c++) offsets[c + 1] = offsets[c] + counts[c];
        int[] cursor = Arrays.copyOf(offsets, offsets.length);
        System.err.printf("assigned counts countMs=%d min=%d max=%d avg=%.1f%n", ms(t2), min(counts), max(counts), n / (double) K);

        long t3 = System.nanoTime();
        short[][] outDims = new short[DIMS][n];
        byte[] labels = new byte[n];
        int[] origIds = new int[n];
        short[][] bboxMin = new short[DIMS][K];
        short[][] bboxMax = new short[DIMS][K];
        for (int d = 0; d < DIMS; d++) {
            Arrays.fill(bboxMin[d], Short.MAX_VALUE);
            Arrays.fill(bboxMax[d], Short.MIN_VALUE);
        }

        for (int i = 0; i < n; i++) {
            int c = nearest(dims, i, centroids);
            int pos = cursor[c]++;
            for (int d = 0; d < DIMS; d++) {
                short v = dims[d][i];
                outDims[d][pos] = v;
                if (v < bboxMin[d][c]) bboxMin[d][c] = v;
                if (v > bboxMax[d][c]) bboxMax[d][c] = v;
            }
            labels[pos] = srcLabels[i];
            origIds[pos] = i;
        }

        for (int c = 0; c < K; c++) {
            if (offsets[c] == offsets[c + 1]) {
                for (int d = 0; d < DIMS; d++) {
                    bboxMin[d][c] = centroids[d][c];
                    bboxMax[d][c] = centroids[d][c];
                }
            }
        }
        System.err.printf("sorted+bbox sortMs=%d%n", ms(t3));

        long t4 = System.nanoTime();
        Files.createDirectories(out.toAbsolutePath().getParent());
        write(out, n, K, offsets, centroids, bboxMin, bboxMax, outDims, labels, origIds);
        System.err.printf("wrote %s sizeMB=%.1f writeMs=%d totalMs=%d%n", out,
                Files.size(out) / 1024.0 / 1024.0, ms(t4), ms(t0));
    }

    private static int[] makeSample(int n, int sample, long seed) {
        int[] idx = new int[sample];
        if (sample == n) {
            for (int i = 0; i < n; i++) idx[i] = i;
            return idx;
        }

        Random rnd = new Random(seed);
        double step = n / (double) sample;
        for (int i = 0; i < sample; i++) {
            int base = (int) (i * step);
            int span = Math.max(1, (int) step);
            idx[i] = Math.min(n - 1, base + rnd.nextInt(span));
        }
        return idx;
    }

    private static short[][] initCentroids(short[][] dims, int[] sampleIdx, int k, long seed) {
        Random rnd = new Random(seed ^ 0x9E3779B97F4A7C15L);
        short[][] c = new short[DIMS][k];
        int first = sampleIdx[rnd.nextInt(sampleIdx.length)];
        copyVector(dims, first, c, 0);

        long[] nearest = new long[sampleIdx.length];
        Arrays.fill(nearest, Long.MAX_VALUE);

        for (int ci = 1; ci < k; ci++) {
            int prev = ci - 1;
            for (int s = 0; s < sampleIdx.length; s++) {
                long d = distToCentroid(dims, sampleIdx[s], c, prev);
                if (d < nearest[s]) nearest[s] = d;
            }
            int bestS = rnd.nextInt(sampleIdx.length);
            long bestD = -1;
            int candidates = Math.min(256, sampleIdx.length);
            for (int t = 0; t < candidates; t++) {
                int s = rnd.nextInt(sampleIdx.length);
                if (nearest[s] > bestD) { bestD = nearest[s]; bestS = s; }
            }
            copyVector(dims, sampleIdx[bestS], c, ci);
        }
        return c;
    }

    private static void runLloyd(short[][] dims, int[] sampleIdx, short[][] centroids, int iters) {
        long[][] sums = new long[DIMS][K];
        int[] counts = new int[K];
        int[] assign = new int[sampleIdx.length];
        Arrays.fill(assign, -1);
        Random rnd = new Random(SEED ^ 0xC0FFEE);

        for (int it = 0; it < iters; it++) {
            for (int d = 0; d < DIMS; d++) Arrays.fill(sums[d], 0L);
            Arrays.fill(counts, 0);
            int changed = 0;
            for (int s = 0; s < sampleIdx.length; s++) {
                int idx = sampleIdx[s];
                int c = nearest(dims, idx, centroids);
                if (assign[s] != c) { assign[s] = c; changed++; }
                counts[c]++;
                for (int d = 0; d < DIMS; d++) sums[d][c] += dims[d][idx];
            }
            for (int c = 0; c < K; c++) {
                if (counts[c] == 0) {
                    copyVector(dims, sampleIdx[rnd.nextInt(sampleIdx.length)], centroids, c);
                } else {
                    for (int d = 0; d < DIMS; d++) centroids[d][c] = (short) Math.round(sums[d][c] / (double) counts[c]);
                }
            }
            System.err.printf("iter=%d changed=%d empty=%d%n", it + 1, changed, empty(counts));
            if (changed == 0) break;
        }
    }

    private static int nearest(short[][] dims, int i, short[][] c) {
        long best = Long.MAX_VALUE;
        int bestC = 0;
        for (int ci = 0; ci < c[0].length; ci++) {
            long d = distToCentroid(dims, i, c, ci);
            if (d < best) { best = d; bestC = ci; }
        }
        return bestC;
    }

    private static long distToCentroid(short[][] dims, int i, short[][] c, int ci) {
        long s = 0;
        int x;
        x = dims[0][i] - c[0][ci]; s += (long) x * x;
        x = dims[1][i] - c[1][ci]; s += (long) x * x;
        x = dims[2][i] - c[2][ci]; s += (long) x * x;
        x = dims[3][i] - c[3][ci]; s += (long) x * x;
        x = dims[4][i] - c[4][ci]; s += (long) x * x;
        x = dims[5][i] - c[5][ci]; s += (long) x * x;
        x = dims[6][i] - c[6][ci]; s += (long) x * x;
        x = dims[7][i] - c[7][ci]; s += (long) x * x;
        x = dims[8][i] - c[8][ci]; s += (long) x * x;
        x = dims[9][i] - c[9][ci]; s += (long) x * x;
        x = dims[10][i] - c[10][ci]; s += (long) x * x;
        x = dims[11][i] - c[11][ci]; s += (long) x * x;
        x = dims[12][i] - c[12][ci]; s += (long) x * x;
        x = dims[13][i] - c[13][ci]; s += (long) x * x;
        return s;
    }

    private static void copyVector(short[][] dims, int i, short[][] c, int ci) {
        for (int d = 0; d < DIMS; d++) c[d][ci] = dims[d][i];
    }

    private static int min(int[] a) { int m = Integer.MAX_VALUE; for (int v : a) if (v < m) m = v; return m; }
    private static int max(int[] a) { int m = Integer.MIN_VALUE; for (int v : a) if (v > m) m = v; return m; }
    private static int empty(int[] a) { int n = 0; for (int v : a) if (v == 0) n++; return n; }
    private static long ms(long t0) { return (System.nanoTime() - t0) / 1_000_000L; }

    private static void write(Path out, int n, int k, int[] offsets, short[][] centroids, short[][] bboxMin,
                              short[][] bboxMax, short[][] dims, byte[] labels, int[] origIds) throws IOException {
        try (DataOutputStream w = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(out), 1 << 20))) {
            writeIntLE(w, MAGIC); writeIntLE(w, VERSION); writeIntLE(w, n); writeIntLE(w, k); writeIntLE(w, DIMS); writeIntLE(w, SCALE);
            for (int v : offsets) writeIntLE(w, v);
            writeShortMatrix(w, centroids, k);
            writeShortMatrix(w, bboxMin, k);
            writeShortMatrix(w, bboxMax, k);
            writeShortMatrix(w, dims, n);
            w.write(labels);
            for (int v : origIds) writeIntLE(w, v);
        }
    }

    private static void writeShortMatrix(DataOutputStream w, short[][] m, int len) throws IOException {
        byte[] buf = new byte[len * 2];
        for (int d = 0; d < DIMS; d++) {
            ByteBuffer bb = ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < len; i++) bb.putShort(m[d][i]);
            w.write(buf);
        }
    }

    private static void writeIntLE(DataOutputStream w, int v) throws IOException {
        w.write(v & 255); w.write((v >>> 8) & 255); w.write((v >>> 16) & 255); w.write((v >>> 24) & 255);
    }
}
