package rinha.knn;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

public final class IvfIndex implements FraudIndex {
    static final int DIMS = 14;
    static final int SCALE = 10_000;
    static final int MAGIC = 0x4956464A;
    static final int VERSION = 6;

    private final int n;
    private final int k;
    private final short[][] dims;
    private final short[] rows;
    private final byte[] labels;
    private final int[] origIds;
    private final int[] offsets;
    private final short[][] bboxMin;
    private final short[][] bboxMax;
    private final short[][] centroids;
    private final short[] bboxMinCM;
    private final short[] bboxMaxCM;
    private final short[] centroidsCM;
    private final int nprobe;
    private final int[] nonEmptyClusters;
    private final boolean repairEnabled;

    private IvfIndex(
            int n,
            int k,
            short[][] dims,
            short[] rows,
            byte[] labels,
            int[] origIds,
            int[] offsets,
            short[][] bboxMin,
            short[][] bboxMax,
            short[][] centroids,
            short[] bboxMinCM,
            short[] bboxMaxCM,
            short[] centroidsCM,
            int nprobe,
            int[] nonEmptyClusters
    ) {
        this.repairEnabled = Boolean.parseBoolean(System.getProperty("ivf.repair", "true"));
        this.n = n;
        this.k = k;
        this.dims = dims;
        this.rows = rows;
        this.labels = labels;
        this.origIds = origIds;
        this.offsets = offsets;
        this.bboxMin = bboxMin;
        this.bboxMax = bboxMax;
        this.centroids = centroids;
        this.bboxMinCM = bboxMinCM;
        this.bboxMaxCM = bboxMaxCM;
        this.centroidsCM = centroidsCM;
        this.nprobe = Math.max(1, Math.min(nprobe, k));
        this.nonEmptyClusters = nonEmptyClusters;
    }

    @Override
    public int size() {
        return n;
    }

    public int clusters() {
        return k;
    }

    public int nonEmptyClusters() {
        return nonEmptyClusters.length;
    }

    public int nprobe() {
        return nprobe;
    }

    public boolean repairEnabled() {
        return repairEnabled;
    }

    public static IvfIndex load(Path path) throws IOException {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(path), 1 << 20))) {
            int magic = readIntLE(in);
            int version = readIntLE(in);
            if (magic != MAGIC || version < 3 || version > VERSION) {
                throw new IOException("bad index.bin: magic/version mismatch");
            }

            int n = readIntLE(in);
            int k = readIntLE(in);
            int dimsN = readIntLE(in);
            int scale = readIntLE(in);
            if (dimsN != DIMS || scale != SCALE) {
                throw new IOException("bad index.bin dims/scale");
            }

            int[] offsets = new int[k + 1];
            for (int i = 0; i <= k; i++) {
                offsets[i] = readIntLE(in);
            }

            short[][] centroids = new short[DIMS][k];
            readShortMatrix(in, centroids, k);

            short[][] bboxMin = new short[DIMS][k];
            readShortMatrix(in, bboxMin, k);

            short[][] bboxMax = new short[DIMS][k];
            readShortMatrix(in, bboxMax, k);

            short[][] dataDims = null;
            short[] rows = null;
            if (version >= 6) {
                rows = readRows(in, n);
            } else {
                dataDims = new short[DIMS][n];
                readShortMatrix(in, dataDims, n);
            }

            byte[] labels = new byte[n];
            in.readFully(labels);

            int[] origIds = new int[n];
            for (int i = 0; i < n; i++) {
                origIds[i] = readIntLE(in);
            }

            short[] centroidsCM = flattenClusterMajor(centroids, k);
            short[] bboxMinCM = flattenClusterMajor(bboxMin, k);
            short[] bboxMaxCM = flattenClusterMajor(bboxMax, k);

            int nprobe = Integer.getInteger("ivf.nprobe", 8);

            int nonEmpty = 0;
            for (int c = 0; c < k; c++) {
                if (offsets[c] != offsets[c + 1]) {
                    nonEmpty++;
                }
            }

            int[] nonEmptyClusters = new int[nonEmpty];
            for (int c = 0, j = 0; c < k; c++) {
                if (offsets[c] != offsets[c + 1]) {
                    nonEmptyClusters[j++] = c;
                }
            }

            return new IvfIndex(
                    n,
                    k,
                    dataDims,
                    rows,
                    labels,
                    origIds,
                    offsets,
                    bboxMin,
                    bboxMax,
                    centroids,
                    bboxMinCM,
                    bboxMaxCM,
                    centroidsCM,
                    nprobe,
                    nonEmptyClusters
            );
        }
    }

    @Override
    public int searchFraudCount(short[] q) {
        if (nprobe == 1) {
            return searchFraudCountNprobe1(q);
        }

        SearchStats stats = STATS.get();
        stats.reset();
        return searchFraudCountGeneric(q, stats);
    }

    private int searchFraudCountNprobe1(short[] q) {
        Top5 top = new Top5();
        int chosen = selectBestCluster(q);
        if (chosen >= 0) {
            scanClusterNoStats(chosen, q, top);
        }

        if (repairEnabled) {
            long worst = top.worstDist();
            for (int idx = 0; idx < nonEmptyClusters.length; idx++) {
                int c = nonEmptyClusters[idx];
                if (c == chosen) {
                    continue;
                }
                if (bboxMayBeat(c, q, worst)) {
                    scanClusterNoStats(c, q, top);
                    worst = top.worstDist();
                }
            }
        }

        return top.frauds(labels);
    }

    private int searchFraudCountGeneric(short[] q, SearchStats stats) {
        Top5 top = new Top5();
        int[] chosen = TopBuffers.clusters.get();
        long[] chosenDist = TopBuffers.distances.get();
        int count = selectProbeClusters(q, chosen, chosenDist);

        for (int i = 0; i < count; i++) {
            scanCluster(chosen[i], q, top, stats, true);
        }

        if (repairEnabled) {
            for (int idx = 0; idx < nonEmptyClusters.length; idx++) {
                int c = nonEmptyClusters[idx];
                if (contains(chosen, count, c)) {
                    continue;
                }
                stats.bboxChecks++;
                if (bboxMayBeat(c, q, top.worstDist())) {
                    scanCluster(c, q, top, stats, false);
                }
            }
        }

        return top.frauds(labels);
    }

    public SearchStats lastStats() {
        return STATS.get();
    }

    private int selectBestCluster(short[] q) {
        int bestC = -1;
        long bestD = Long.MAX_VALUE;

        for (int idx = 0; idx < nonEmptyClusters.length; idx++) {
            int c = nonEmptyClusters[idx];
            long d = centroidDistance(c, q);
            if (d < bestD) {
                bestD = d;
                bestC = c;
            }
        }

        return bestC;
    }

    private int selectProbeClusters(short[] q, int[] bestC, long[] bestD) {
        int count = nprobe;
        Arrays.fill(bestC, 0, count, -1);
        Arrays.fill(bestD, 0, count, Long.MAX_VALUE);

        for (int idx = 0; idx < nonEmptyClusters.length; idx++) {
            int c = nonEmptyClusters[idx];
            long d = centroidDistance(c, q);
            int pos = -1;

            for (int i = 0; i < count; i++) {
                if (d < bestD[i]) {
                    pos = i;
                    break;
                }
            }

            if (pos >= 0) {
                for (int j = count - 1; j > pos; j--) {
                    bestD[j] = bestD[j - 1];
                    bestC[j] = bestC[j - 1];
                }
                bestD[pos] = d;
                bestC[pos] = c;
            }
        }

        int valid = 0;
        for (int i = 0; i < count; i++) {
            if (bestC[i] >= 0) {
                valid++;
            }
        }
        return valid;
    }

    private long centroidDistance(int c, short[] q) {
        final int base = c * DIMS;
        final short[] cm = centroidsCM;

        long s = 0;
        int x;

        x = cm[base] - q[0]; s += (long) x * x;
        x = cm[base + 1] - q[1]; s += (long) x * x;
        x = cm[base + 2] - q[2]; s += (long) x * x;
        x = cm[base + 3] - q[3]; s += (long) x * x;
        x = cm[base + 4] - q[4]; s += (long) x * x;
        x = cm[base + 5] - q[5]; s += (long) x * x;
        x = cm[base + 6] - q[6]; s += (long) x * x;
        x = cm[base + 7] - q[7]; s += (long) x * x;
        x = cm[base + 8] - q[8]; s += (long) x * x;
        x = cm[base + 9] - q[9]; s += (long) x * x;
        x = cm[base + 10] - q[10]; s += (long) x * x;
        x = cm[base + 11] - q[11]; s += (long) x * x;
        x = cm[base + 12] - q[12]; s += (long) x * x;
        x = cm[base + 13] - q[13]; s += (long) x * x;

        return s;
    }

    private boolean bboxMayBeat(int c, short[] q, long worst) {
        final int base = c * DIMS;
        final short[] mn = bboxMinCM;
        final short[] mx = bboxMaxCM;

        long s = 0;
        int v;
        int diff;

        v = q[0]; diff = v < mn[base] ? mn[base] - v : (v > mx[base] ? v - mx[base] : 0); s += (long) diff * diff; if (s > worst) return false;
        v = q[1]; diff = v < mn[base + 1] ? mn[base + 1] - v : (v > mx[base + 1] ? v - mx[base + 1] : 0); s += (long) diff * diff; if (s > worst) return false;
        v = q[2]; diff = v < mn[base + 2] ? mn[base + 2] - v : (v > mx[base + 2] ? v - mx[base + 2] : 0); s += (long) diff * diff; if (s > worst) return false;
        v = q[3]; diff = v < mn[base + 3] ? mn[base + 3] - v : (v > mx[base + 3] ? v - mx[base + 3] : 0); s += (long) diff * diff; if (s > worst) return false;
        v = q[4]; diff = v < mn[base + 4] ? mn[base + 4] - v : (v > mx[base + 4] ? v - mx[base + 4] : 0); s += (long) diff * diff; if (s > worst) return false;
        v = q[5]; diff = v < mn[base + 5] ? mn[base + 5] - v : (v > mx[base + 5] ? v - mx[base + 5] : 0); s += (long) diff * diff; if (s > worst) return false;
        v = q[6]; diff = v < mn[base + 6] ? mn[base + 6] - v : (v > mx[base + 6] ? v - mx[base + 6] : 0); s += (long) diff * diff; if (s > worst) return false;
        v = q[7]; diff = v < mn[base + 7] ? mn[base + 7] - v : (v > mx[base + 7] ? v - mx[base + 7] : 0); s += (long) diff * diff; if (s > worst) return false;
        v = q[8]; diff = v < mn[base + 8] ? mn[base + 8] - v : (v > mx[base + 8] ? v - mx[base + 8] : 0); s += (long) diff * diff; if (s > worst) return false;
        v = q[9]; diff = v < mn[base + 9] ? mn[base + 9] - v : (v > mx[base + 9] ? v - mx[base + 9] : 0); s += (long) diff * diff; if (s > worst) return false;
        v = q[10]; diff = v < mn[base + 10] ? mn[base + 10] - v : (v > mx[base + 10] ? v - mx[base + 10] : 0); s += (long) diff * diff; if (s > worst) return false;
        v = q[11]; diff = v < mn[base + 11] ? mn[base + 11] - v : (v > mx[base + 11] ? v - mx[base + 11] : 0); s += (long) diff * diff; if (s > worst) return false;
        v = q[12]; diff = v < mn[base + 12] ? mn[base + 12] - v : (v > mx[base + 12] ? v - mx[base + 12] : 0); s += (long) diff * diff; if (s > worst) return false;
        v = q[13]; diff = v < mn[base + 13] ? mn[base + 13] - v : (v > mx[base + 13] ? v - mx[base + 13] : 0); s += (long) diff * diff;

        return s <= worst;
    }

    private void scanCluster(int c, short[] q, Top5 top, SearchStats stats, boolean probe) {
        int start = offsets[c];
        int end = offsets[c + 1];
        int len = end - start;

        if (len <= 0) {
            return;
        }

        stats.scannedClusters++;
        stats.scannedVectors += len;

        if (probe) {
            stats.probeClusters++;
            stats.probeVectors += len;
        } else {
            stats.repairClusters++;
            stats.repairVectors += len;
        }

        if (rows != null) {
            scanClusterRows(start, end, q, top);
        } else {
            scanClusterSoaScalar(start, end, q, top);
        }
    }
    private void scanClusterNoStats(int c, short[] q, Top5 top) {
        int start = offsets[c];
        int end = offsets[c + 1];
        if (start == end) {
            return;
        }
        if (rows != null) {
            scanClusterRows(start, end, q, top);
        } else {
            scanClusterSoaScalar(start, end, q, top);
        }
    }


    private void scanClusterSoaScalar(int start, int end, short[] q, Top5 top) {
        final short[] d0s = dims[0], d1s = dims[1], d2s = dims[2], d3s = dims[3], d4s = dims[4], d5s = dims[5], d6s = dims[6];
        final short[] d7s = dims[7], d8s = dims[8], d9s = dims[9], d10s = dims[10], d11s = dims[11], d12s = dims[12], d13s = dims[13];
        final int q0 = q[0], q1 = q[1], q2 = q[2], q3 = q[3], q4 = q[4], q5 = q[5], q6 = q[6];
        final int q7 = q[7], q8 = q[8], q9 = q[9], q10 = q[10], q11 = q[11], q12 = q[12], q13 = q[13];

        long worst = top.d4;

        for (int i = start; i < end; i++) {
            long dist = sq(d0s[i] - q0); if (dist > worst) continue;
            dist += sq(d1s[i] - q1); if (dist > worst) continue;
            dist += sq(d2s[i] - q2); if (dist > worst) continue;
            dist += sq(d3s[i] - q3); if (dist > worst) continue;
            dist += sq(d4s[i] - q4); if (dist > worst) continue;
            dist += sq(d5s[i] - q5); if (dist > worst) continue;
            dist += sq(d6s[i] - q6); if (dist > worst) continue;
            dist += sq(d7s[i] - q7); if (dist > worst) continue;
            dist += sq(d8s[i] - q8); if (dist > worst) continue;
            dist += sq(d9s[i] - q9); if (dist > worst) continue;
            dist += sq(d10s[i] - q10); if (dist > worst) continue;
            dist += sq(d11s[i] - q11); if (dist > worst) continue;
            dist += sq(d12s[i] - q12); if (dist > worst) continue;
            dist += sq(d13s[i] - q13); if (dist > worst) continue;
            top.add(dist, i, origIds[i]);
            worst = top.d4;
        }
    }

    private void scanClusterRows(int start, int end, short[] q, Top5 top) {
        final short[] r = rows;
        final int q0 = q[0], q1 = q[1], q2 = q[2], q3 = q[3], q4 = q[4], q5 = q[5], q6 = q[6];
        final int q7 = q[7], q8 = q[8], q9 = q[9], q10 = q[10], q11 = q[11], q12 = q[12], q13 = q[13];

        long worst = top.d4;

        for (int i = start; i < end; i++) {
            int b = i * DIMS;

            long dist = sq(r[b] - q0); if (dist > worst) continue;
            dist += sq(r[b + 1] - q1); if (dist > worst) continue;
            dist += sq(r[b + 2] - q2); if (dist > worst) continue;
            dist += sq(r[b + 3] - q3); if (dist > worst) continue;
            dist += sq(r[b + 4] - q4); if (dist > worst) continue;
            dist += sq(r[b + 5] - q5); if (dist > worst) continue;
            dist += sq(r[b + 6] - q6); if (dist > worst) continue;
            dist += sq(r[b + 7] - q7); if (dist > worst) continue;
            dist += sq(r[b + 8] - q8); if (dist > worst) continue;
            dist += sq(r[b + 9] - q9); if (dist > worst) continue;
            dist += sq(r[b + 10] - q10); if (dist > worst) continue;
            dist += sq(r[b + 11] - q11); if (dist > worst) continue;
            dist += sq(r[b + 12] - q12); if (dist > worst) continue;
            dist += sq(r[b + 13] - q13); if (dist > worst) continue;

            top.add(dist, i, origIds[i]);
            worst = top.d4;
        }
    }

    private static boolean contains(int[] a, int len, int v) {
        for (int i = 0; i < len; i++) {
            if (a[i] == v) {
                return true;
            }
        }
        return false;
    }

    private static long sq(int x) {
        return (long) x * x;
    }

    private static short[] flattenClusterMajor(short[][] src, int k) {
        short[] out = new short[k * DIMS];

        for (int c = 0; c < k; c++) {
            int base = c * DIMS;
            out[base] = src[0][c];
            out[base + 1] = src[1][c];
            out[base + 2] = src[2][c];
            out[base + 3] = src[3][c];
            out[base + 4] = src[4][c];
            out[base + 5] = src[5][c];
            out[base + 6] = src[6][c];
            out[base + 7] = src[7][c];
            out[base + 8] = src[8][c];
            out[base + 9] = src[9][c];
            out[base + 10] = src[10][c];
            out[base + 11] = src[11][c];
            out[base + 12] = src[12][c];
            out[base + 13] = src[13][c];
        }

        return out;
    }

    private static int readIntLE(DataInputStream in) throws IOException {
        int b0 = in.read();
        int b1 = in.read();
        int b2 = in.read();
        int b3 = in.read();

        if ((b0 | b1 | b2 | b3) < 0) {
            throw new EOFException();
        }

        return (b0 & 255)
                | ((b1 & 255) << 8)
                | ((b2 & 255) << 16)
                | ((b3 & 255) << 24);
    }

    private static short[] readRows(DataInputStream in, int n) throws IOException {
        short[] rows = new short[n * DIMS];
        final int shortsPerChunk = 1 << 16;
        byte[] buf = new byte[shortsPerChunk * 2];

        int off = 0;
        int total = rows.length;

        while (off < total) {
            int cnt = Math.min(shortsPerChunk, total - off);
            in.readFully(buf, 0, cnt * 2);

            ByteBuffer bb = ByteBuffer.wrap(buf, 0, cnt * 2).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < cnt; i++) {
                rows[off + i] = bb.getShort();
            }

            off += cnt;
        }

        return rows;
    }

    private static void readShortMatrix(DataInputStream in, short[][] dst, int len) throws IOException {
        byte[] buf = new byte[len * 2];

        for (int d = 0; d < DIMS; d++) {
            in.readFully(buf);
            ByteBuffer bb = ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN);

            for (int i = 0; i < len; i++) {
                dst[d][i] = bb.getShort();
            }
        }
    }

    private static final ThreadLocal<SearchStats> STATS = ThreadLocal.withInitial(SearchStats::new);

    public static final class SearchStats {
        public int scannedClusters;
        public int scannedVectors;
        public int probeClusters;
        public int probeVectors;
        public int repairClusters;
        public int repairVectors;
        public int bboxChecks;

        void reset() {
            scannedClusters = 0;
            scannedVectors = 0;
            probeClusters = 0;
            probeVectors = 0;
            repairClusters = 0;
            repairVectors = 0;
            bboxChecks = 0;
        }
    }

    private static final class TopBuffers {
        static final ThreadLocal<int[]> clusters = ThreadLocal.withInitial(() -> new int[Math.max(1, Integer.getInteger("ivf.nprobe", 8))]);
        static final ThreadLocal<long[]> distances = ThreadLocal.withInitial(() -> new long[Math.max(1, Integer.getInteger("ivf.nprobe", 8))]);
    }

    private static final class Top5 {
        long d0 = Long.MAX_VALUE;
        long d1 = Long.MAX_VALUE;
        long d2 = Long.MAX_VALUE;
        long d3 = Long.MAX_VALUE;
        long d4 = Long.MAX_VALUE;

        int pos0 = -1;
        int pos1 = -1;
        int pos2 = -1;
        int pos3 = -1;
        int pos4 = -1;

        int id0 = Integer.MAX_VALUE;
        int id1 = Integer.MAX_VALUE;
        int id2 = Integer.MAX_VALUE;
        int id3 = Integer.MAX_VALUE;
        int id4 = Integer.MAX_VALUE;

        long worstDist() {
            return d4;
        }

        void add(long d, int pos, int orig) {
            if (!beats(d, orig, d4, id4)) {
                return;
            }

            if (beats(d, orig, d3, id3)) {
                d4 = d3;
                pos4 = pos3;
                id4 = id3;

                if (beats(d, orig, d2, id2)) {
                    d3 = d2;
                    pos3 = pos2;
                    id3 = id2;

                    if (beats(d, orig, d1, id1)) {
                        d2 = d1;
                        pos2 = pos1;
                        id2 = id1;

                        if (beats(d, orig, d0, id0)) {
                            d1 = d0;
                            pos1 = pos0;
                            id1 = id0;
                            d0 = d;
                            pos0 = pos;
                            id0 = orig;
                        } else {
                            d1 = d;
                            pos1 = pos;
                            id1 = orig;
                        }
                    } else {
                        d2 = d;
                        pos2 = pos;
                        id2 = orig;
                    }
                } else {
                    d3 = d;
                    pos3 = pos;
                    id3 = orig;
                }
            } else {
                d4 = d;
                pos4 = pos;
                id4 = orig;
            }
        }

        int frauds(byte[] labels) {
            int f = 0;

            if (pos0 >= 0) {
                f += labels[pos0];
            }
            if (pos1 >= 0) {
                f += labels[pos1];
            }
            if (pos2 >= 0) {
                f += labels[pos2];
            }
            if (pos3 >= 0) {
                f += labels[pos3];
            }
            if (pos4 >= 0) {
                f += labels[pos4];
            }

            return f;
        }

        private static boolean beats(long d, int id, long curD, int curId) {
            return d < curD || (d == curD && id < curId);
        }
    }
}
