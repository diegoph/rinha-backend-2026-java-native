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

public final class BuildIndex {
    private static final int DIMS = 14;
    private static final int SCALE = 10_000;
    private static final int MAGIC = 0x4956464A;
    private static final int VERSION = 5;



    private static final int LAST_BINS = 2;
    private static final int MCC_BINS = 11;
    private static final int AMOUNT_BINS = Integer.getInteger("grid.amountBins", 16);
    private static final int RATIO_BINS = Integer.getInteger("grid.ratioBins", 16);
    private static final int HOUR_BINS = Integer.getInteger("grid.hourBins", 8);
    private static final int DAY_BINS = Integer.getInteger("grid.dayBins", 1);
    private static final int TX_BINS = Integer.getInteger("grid.txBins", 1);
    private static final boolean FLAGS = Boolean.parseBoolean(System.getProperty("grid.flags", "false"));
    private static final int ONLINE_BINS = FLAGS ? 2 : 1;
    private static final int PRESENT_BINS = FLAGS ? 2 : 1;
    private static final int UNKNOWN_BINS = FLAGS ? 2 : 1;
    private static final int RAW_K = checkedRawK();

    public static void main(String[] args) throws Exception {
        Path refs = args.length > 0 ? Path.of(args[0]) : Path.of("resources/references.json.gz");
        Path out = args.length > 1 ? Path.of(args[1]) : Path.of("resources/index.bin");

        long t0 = System.nanoTime();
        ReferenceIndex src = ReferenceIndex.load(refs);
        int n = src.size();
        System.err.printf("loaded refs=%s n=%d loadMs=%d%n", refs, n, ms(t0));
        System.err.printf("grid rawK=%d amount=%d ratio=%d hour=%d day=%d tx=%d flags=%s%n",
                RAW_K, AMOUNT_BINS, RATIO_BINS, HOUR_BINS, DAY_BINS, TX_BINS, FLAGS);

        short[][] dims = new short[DIMS][];
        for (int d = 0; d < DIMS; d++) dims[d] = src.dim(d);
        byte[] srcLabels = src.labels();

        long t1 = System.nanoTime();
        int[] rawCounts = new int[RAW_K];
        for (int i = 0; i < n; i++) rawCounts[clusterKey(dims, i)]++;
        int compactK = nonEmpty(rawCounts);
        int[] rawToCompact = new int[RAW_K];
        Arrays.fill(rawToCompact, -1);
        int[] counts = new int[compactK];
        for (int raw = 0, c = 0; raw < RAW_K; raw++) {
            int cnt = rawCounts[raw];
            if (cnt != 0) {
                rawToCompact[raw] = c;
                counts[c] = cnt;
                c++;
            }
        }
        int[] offsets = new int[compactK + 1];
        for (int c = 0; c < compactK; c++) offsets[c + 1] = offsets[c] + counts[c];
        int[] cursor = Arrays.copyOf(offsets, offsets.length);
        System.err.printf("counted clusters compactK=%d countMs=%d%n", compactK, ms(t1));

        short[][] outDims = new short[DIMS][n];
        byte[] labels = new byte[n];
        int[] origIds = new int[n];
        short[][] bboxMin = new short[DIMS][compactK];
        short[][] bboxMax = new short[DIMS][compactK];
        short[][] centroids = new short[DIMS][compactK];
        for (int d = 0; d < DIMS; d++) {
            Arrays.fill(bboxMin[d], Short.MAX_VALUE);
            Arrays.fill(bboxMax[d], Short.MIN_VALUE);
        }

        long t2 = System.nanoTime();
        for (int i = 0; i < n; i++) {
            int raw = clusterKey(dims, i);
            int c = rawToCompact[raw];
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

        for (int c = 0; c < compactK; c++) {
            for (int d = 0; d < DIMS; d++) {
                centroids[d][c] = (short) ((bboxMin[d][c] + bboxMax[d][c]) / 2);
            }
        }
        System.err.printf("sorted+bbox sortMs=%d%n", ms(t2));

        long t3 = System.nanoTime();
        Files.createDirectories(out.toAbsolutePath().getParent());
        write(out, n, compactK, offsets, centroids, bboxMin, bboxMax, outDims, labels, origIds);
        System.err.printf("wrote %s sizeMB=%.1f writeMs=%d totalMs=%d%n", out,
                Files.size(out) / 1024.0 / 1024.0, ms(t3), ms(t0));
    }

    static int clusterKey(short[][] dims, int i) {
        int key = dims[5][i] < 0 ? 0 : 1;
        key = key * MCC_BINS + binNonNeg(dims[12][i], MCC_BINS);
        key = key * AMOUNT_BINS + binNonNeg(dims[0][i], AMOUNT_BINS);
        key = key * RATIO_BINS + binNonNeg(dims[2][i], RATIO_BINS);
        key = key * HOUR_BINS + binNonNeg(dims[3][i], HOUR_BINS);
        if (DAY_BINS > 1) key = key * DAY_BINS + binNonNeg(dims[4][i], DAY_BINS);
        if (TX_BINS > 1) key = key * TX_BINS + binNonNeg(dims[8][i], TX_BINS);
        if (ONLINE_BINS > 1) key = key * ONLINE_BINS + boolBin(dims[9][i]);
        if (PRESENT_BINS > 1) key = key * PRESENT_BINS + boolBin(dims[10][i]);
        if (UNKNOWN_BINS > 1) key = key * UNKNOWN_BINS + boolBin(dims[11][i]);
        return key;
    }

    private static int boolBin(short v) {
        return v >= (SCALE / 2) ? 1 : 0;
    }

    private static int binNonNeg(short v, int bins) {
        int x = v;
        if (x < 0) x = 0;
        if (x > SCALE) x = SCALE;
        int b = (int) (((long) x * bins) / (SCALE + 1L));
        return b >= bins ? bins - 1 : b;
    }

    private static int checkedRawK() {
        long k = 1L;
        k *= LAST_BINS;
        k *= MCC_BINS;
        k *= AMOUNT_BINS;
        k *= RATIO_BINS;
        k *= HOUR_BINS;
        if (DAY_BINS > 1) k *= DAY_BINS;
        if (TX_BINS > 1) k *= TX_BINS;
        if (FLAGS) k *= 8L;
        if (k > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("grid too large: rawK=" + k);
        }
        return (int) k;
    }

    private static int nonEmpty(int[] counts) {
        int n = 0;
        for (int c : counts) if (c != 0) n++;
        return n;
    }

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

    private static void writeRows(DataOutputStream w, short[][] m, int len) throws IOException {


        final int rowsPerChunk = 4096;
        byte[] buf = new byte[rowsPerChunk * DIMS * 2];
        for (int base = 0; base < len; base += rowsPerChunk) {
            int rows = Math.min(rowsPerChunk, len - base);
            ByteBuffer bb = ByteBuffer.wrap(buf, 0, rows * DIMS * 2).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < rows; i++) {
                int pos = base + i;
                bb.putShort(m[0][pos]);
                bb.putShort(m[1][pos]);
                bb.putShort(m[2][pos]);
                bb.putShort(m[3][pos]);
                bb.putShort(m[4][pos]);
                bb.putShort(m[5][pos]);
                bb.putShort(m[6][pos]);
                bb.putShort(m[7][pos]);
                bb.putShort(m[8][pos]);
                bb.putShort(m[9][pos]);
                bb.putShort(m[10][pos]);
                bb.putShort(m[11][pos]);
                bb.putShort(m[12][pos]);
                bb.putShort(m[13][pos]);
            }
            w.write(buf, 0, rows * DIMS * 2);
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
