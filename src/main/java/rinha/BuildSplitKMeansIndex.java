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


public final class BuildSplitKMeansIndex {
    private static final int DIMS = 14;
    private static final int SCALE = 10_000;
    private static final int MAGIC = 0x4956464A;
    private static final int VERSION = 5;

    private static final int BASE_K = Integer.getInteger("kmeans.k", 256);
    private static final int SAMPLE = Integer.getInteger("kmeans.sample", 131_072);
    private static final int ITERS = Integer.getInteger("kmeans.iters", 25);
    private static final long SEED = Long.getLong("kmeans.seed", 8587310L);


    private static final int SPLIT_MAX = Integer.getInteger("split.max", 16_000);
    private static final int SPLIT_MAX_PARTS = Integer.getInteger("split.maxParts", 8);
    private static final boolean SPLIT_ENABLED = Boolean.parseBoolean(System.getProperty("split.enabled", "true"));

    private static final int VALUE_OFFSET = 10_000;
    private static final int VALUE_RANGE = 20_001;

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
        System.err.printf("kmeans baseK=%d sample=%d iters=%d seed=%d splitEnabled=%s splitMax=%d splitMaxParts=%d%n",
                BASE_K, Math.min(SAMPLE, n), ITERS, SEED, SPLIT_ENABLED, SPLIT_MAX, SPLIT_MAX_PARTS);

        long t1 = System.nanoTime();
        int[] sampleIdx = makeSample(n, Math.min(SAMPLE, n), SEED);
        short[][] baseCentroids = initCentroids(dims, sampleIdx, BASE_K, SEED);
        runLloyd(dims, sampleIdx, baseCentroids, ITERS);
        System.err.printf("trained centroids trainMs=%d%n", ms(t1));

        long t2 = System.nanoTime();
        int[] baseAssign = new int[n];
        int[] baseCounts = new int[BASE_K];
        for (int i = 0; i < n; i++) {
            int c = nearest(dims, i, baseCentroids);
            baseAssign[i] = c;
            baseCounts[c]++;
        }
        int[] baseOffsets = new int[BASE_K + 1];
        for (int c = 0; c < BASE_K; c++) baseOffsets[c + 1] = baseOffsets[c] + baseCounts[c];
        int[] cursor = Arrays.copyOf(baseOffsets, baseOffsets.length);
        int[] order = new int[n];
        for (int i = 0; i < n; i++) order[cursor[baseAssign[i]]++] = i;
        System.err.printf("assigned base countMs=%d min=%d max=%d avg=%.1f%n", ms(t2), min(baseCounts), max(baseCounts), n / (double) BASE_K);

        int[] partsPerBase = new int[BASE_K];
        int finalK = 0;
        int maxBase = 0;
        for (int c = 0; c < BASE_K; c++) {
            int len = baseOffsets[c + 1] - baseOffsets[c];
            int parts = 1;
            if (SPLIT_ENABLED && len > SPLIT_MAX) {
                parts = Math.min(SPLIT_MAX_PARTS, (len + SPLIT_MAX - 1) / SPLIT_MAX);
            }
            partsPerBase[c] = parts;
            finalK += parts;
            if (len > maxBase) maxBase = len;
        }
        System.err.printf("split plan finalK=%d baseK=%d maxBaseCluster=%d%n", finalK, BASE_K, maxBase);

        long t3 = System.nanoTime();
        int[] offsets = new int[finalK + 1];
        short[][] centroids = new short[DIMS][finalK];
        short[][] bboxMin = new short[DIMS][finalK];
        short[][] bboxMax = new short[DIMS][finalK];
        short[][] outDims = new short[DIMS][n];
        byte[] labels = new byte[n];
        int[] origIds = new int[n];
        for (int d = 0; d < DIMS; d++) {
            Arrays.fill(bboxMin[d], Short.MAX_VALUE);
            Arrays.fill(bboxMax[d], Short.MIN_VALUE);
        }

        int maxClusterLen = maxBase;
        int[] tmp = new int[maxClusterLen];
        int[] counts = new int[VALUE_RANGE];
        int[] pos = new int[VALUE_RANGE];
        long[] sums = new long[DIMS];

        int outCluster = 0;
        int outPos = 0;
        int maxFinal = 0;
        for (int bc = 0; bc < BASE_K; bc++) {
            int start = baseOffsets[bc];
            int end = baseOffsets[bc + 1];
            int len = end - start;
            int parts = partsPerBase[bc];
            if (len == 0) continue;

            int splitDim = chooseSplitDim(dims, order, start, end);
            if (parts > 1) {
                countingSortByDim(dims[splitDim], order, start, end, tmp, counts, pos);

                System.arraycopy(tmp, 0, order, start, len);
            }

            for (int p = 0; p < parts; p++) {
                int s = start + (int) ((long) len * p / parts);
                int e = start + (int) ((long) len * (p + 1) / parts);
                offsets[outCluster] = outPos;
                Arrays.fill(sums, 0L);

                for (int j = s; j < e; j++) {
                    int orig = order[j];
                    for (int d = 0; d < DIMS; d++) {
                        short v = dims[d][orig];
                        outDims[d][outPos] = v;
                        sums[d] += v;
                        if (v < bboxMin[d][outCluster]) bboxMin[d][outCluster] = v;
                        if (v > bboxMax[d][outCluster]) bboxMax[d][outCluster] = v;
                    }
                    labels[outPos] = srcLabels[orig];
                    origIds[outPos] = orig;
                    outPos++;
                }
                int partLen = e - s;
                if (partLen > maxFinal) maxFinal = partLen;
                if (partLen == 0) {
                    for (int d = 0; d < DIMS; d++) {
                        centroids[d][outCluster] = baseCentroids[d][bc];
                        bboxMin[d][outCluster] = baseCentroids[d][bc];
                        bboxMax[d][outCluster] = baseCentroids[d][bc];
                    }
                } else {
                    for (int d = 0; d < DIMS; d++) {
                        centroids[d][outCluster] = (short) Math.round(sums[d] / (double) partLen);
                    }
                }
                outCluster++;
            }
        }
        offsets[finalK] = outPos;
        if (outPos != n || outCluster != finalK) {
            throw new IllegalStateException("bad split write outPos=" + outPos + " outCluster=" + outCluster + " finalK=" + finalK);
        }
        System.err.printf("split+sorted+bbox splitMs=%d finalK=%d maxFinalCluster=%d%n", ms(t3), finalK, maxFinal);

        long t4 = System.nanoTime();
        Files.createDirectories(out.toAbsolutePath().getParent());
        write(out, n, finalK, offsets, centroids, bboxMin, bboxMax, outDims, labels, origIds);
        System.err.printf("wrote %s sizeMB=%.1f writeMs=%d totalMs=%d%n", out,
                Files.size(out) / 1024.0 / 1024.0, ms(t4), ms(t0));
    }

    private static int chooseSplitDim(short[][] dims, int[] order, int start, int end) {
        int bestD = 0;
        int bestRange = -1;
        for (int d = 0; d < DIMS; d++) {
            int min = Integer.MAX_VALUE, max = Integer.MIN_VALUE;
            short[] arr = dims[d];
            for (int i = start; i < end; i++) {
                int v = arr[order[i]];
                if (v < min) min = v;
                if (v > max) max = v;
            }
            int range = max - min;
            if (range > bestRange) { bestRange = range; bestD = d; }
        }
        return bestD;
    }

    private static void countingSortByDim(short[] dim, int[] order, int start, int end, int[] tmp, int[] counts, int[] pos) {
        Arrays.fill(counts, 0);
        for (int i = start; i < end; i++) counts[dim[order[i]] + VALUE_OFFSET]++;
        int running = 0;
        for (int v = 0; v < VALUE_RANGE; v++) {
            pos[v] = running;
            running += counts[v];
        }
        for (int i = start; i < end; i++) {
            int idx = order[i];
            int key = dim[idx] + VALUE_OFFSET;
            tmp[pos[key]++] = idx;
        }
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
        long[][] sums = new long[DIMS][BASE_K];
        int[] counts = new int[BASE_K];
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
            for (int c = 0; c < BASE_K; c++) {
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
