package rinha.knn;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.zip.GZIPInputStream;

public final class ReferenceIndex implements FraudIndex {
    private static final int DIMS = 14;
    private static final int SCALE = 10_000;

    private final int n;
    private final short[][] dims;
    private final byte[] labels;

    private ReferenceIndex(int n, short[][] dims, byte[] labels) {
        this.n = n;
        this.dims = dims;
        this.labels = labels;
    }

    @Override
    public int size() {
        return n;
    }

    public short[] dim(int d) {
        return dims[d];
    }

    public byte[] labels() {
        return labels;
    }

    public static ReferenceIndex load(Path path) throws IOException {
        boolean gz = path.toString().endsWith(".gz");
        int initialCapacity = gz ? 3_000_000 : 1024;
        Builder b = new Builder(initialCapacity, !gz);

        try (InputStream file = Files.newInputStream(path);
             InputStream maybeGz = gz ? new GZIPInputStream(file, 1 << 16) : file;
             BufferedInputStream in = new BufferedInputStream(maybeGz, 1 << 16)) {
            parseReferences(in, b);
        }
        return b.build();
    }



    @Override
    public int searchFraudCount(short[] q) {
        final short[] d0s = dims[0], d1s = dims[1], d2s = dims[2], d3s = dims[3], d4s = dims[4], d5s = dims[5], d6s = dims[6];
        final short[] d7s = dims[7], d8s = dims[8], d9s = dims[9], d10s = dims[10], d11s = dims[11], d12s = dims[12], d13s = dims[13];
        final int q0 = q[0], q1 = q[1], q2 = q[2], q3 = q[3], q4 = q[4], q5 = q[5], q6 = q[6];
        final int q7 = q[7], q8 = q[8], q9 = q[9], q10 = q[10], q11 = q[11], q12 = q[12], q13 = q[13];

        long best0 = Long.MAX_VALUE, best1 = Long.MAX_VALUE, best2 = Long.MAX_VALUE, best3 = Long.MAX_VALUE, best4 = Long.MAX_VALUE;
        int id0 = Integer.MAX_VALUE, id1 = Integer.MAX_VALUE, id2 = Integer.MAX_VALUE, id3 = Integer.MAX_VALUE, id4 = Integer.MAX_VALUE;

        for (int i = 0; i < n; i++) {
            long dist = sq(d0s[i] - q0);
            if (dist > best4) continue;
            dist += sq(d1s[i] - q1);
            if (dist > best4) continue;
            dist += sq(d2s[i] - q2);
            if (dist > best4) continue;
            dist += sq(d3s[i] - q3);
            if (dist > best4) continue;
            dist += sq(d4s[i] - q4);
            if (dist > best4) continue;
            dist += sq(d5s[i] - q5);
            if (dist > best4) continue;
            dist += sq(d6s[i] - q6);
            if (dist > best4) continue;
            dist += sq(d7s[i] - q7);
            if (dist > best4) continue;
            dist += sq(d8s[i] - q8);
            if (dist > best4) continue;
            dist += sq(d9s[i] - q9);
            if (dist > best4) continue;
            dist += sq(d10s[i] - q10);
            if (dist > best4) continue;
            dist += sq(d11s[i] - q11);
            if (dist > best4) continue;
            dist += sq(d12s[i] - q12);
            if (dist > best4) continue;
            dist += sq(d13s[i] - q13);
            if (dist > best4) continue;
            if (!beats(dist, i, best4, id4)) continue;

            if (beats(dist, i, best3, id3)) {
                best4 = best3; id4 = id3;
                if (beats(dist, i, best2, id2)) {
                    best3 = best2; id3 = id2;
                    if (beats(dist, i, best1, id1)) {
                        best2 = best1; id2 = id1;
                        if (beats(dist, i, best0, id0)) {
                            best1 = best0; id1 = id0;
                            best0 = dist; id0 = i;
                        } else {
                            best1 = dist; id1 = i;
                        }
                    } else {
                        best2 = dist; id2 = i;
                    }
                } else {
                    best3 = dist; id3 = i;
                }
            } else {
                best4 = dist; id4 = i;
            }
        }

        int frauds = 0;
        if (id0 < n) frauds += labels[id0];
        if (id1 < n) frauds += labels[id1];
        if (id2 < n) frauds += labels[id2];
        if (id3 < n) frauds += labels[id3];
        if (id4 < n) frauds += labels[id4];
        return frauds;
    }

    private static long sq(int x) {
        return (long) x * x;
    }

    private static boolean beats(long d, int id, long curD, int curId) {
        return d < curD || (d == curD && id < curId);
    }

    private static void parseReferences(BufferedInputStream in, Builder b) throws IOException {
        byte[] vectorToken = "\"vector\"".getBytes(java.nio.charset.StandardCharsets.US_ASCII);
        byte[] labelToken = "\"label\"".getBytes(java.nio.charset.StandardCharsets.US_ASCII);
        short[] v = new short[DIMS];

        while (findToken(in, vectorToken)) {
            skipUntil(in, '[');
            for (int i = 0; i < DIMS; i++) {
                v[i] = quantize(readNumber(in));
            }

            if (!findToken(in, labelToken)) break;
            skipUntil(in, ':');
            skipUntil(in, '"');
            int first = in.read();
            byte label = (byte) (first == 'f' ? 1 : 0);
            skipUntil(in, '"');

            b.add(v, label);
        }
    }

    private static boolean findToken(BufferedInputStream in, byte[] token) throws IOException {
        int matched = 0;
        int c;
        while ((c = in.read()) >= 0) {
            if (c == token[matched]) {
                matched++;
                if (matched == token.length) return true;
            } else {
                matched = c == token[0] ? 1 : 0;
            }
        }
        return false;
    }

    private static void skipUntil(BufferedInputStream in, int target) throws IOException {
        int c;
        while ((c = in.read()) >= 0) {
            if (c == target) return;
        }
        throw new IOException("unexpected EOF");
    }

    private static double readNumber(BufferedInputStream in) throws IOException {
        int c;
        do {
            c = in.read();
            if (c < 0) throw new IOException("unexpected EOF");
        } while (c != '-' && (c < '0' || c > '9'));

        boolean neg = false;
        if (c == '-') {
            neg = true;
            c = in.read();
        }

        long intPart = 0;
        while (c >= '0' && c <= '9') {
            intPart = intPart * 10 + (c - '0');
            c = in.read();
        }

        double v = intPart;
        if (c == '.') {
            double div = 10.0;
            c = in.read();
            while (c >= '0' && c <= '9') {
                v += (c - '0') / div;
                div *= 10.0;
                c = in.read();
            }
        }
        return neg ? -v : v;
    }

    private static short quantize(double v) {
        return (short) Math.round(v * SCALE);
    }

    private static final class Builder {
        private short[][] dims;
        private byte[] labels;
        private int n;
        private final boolean grow;

        Builder(int capacity, boolean grow) {
            this.grow = grow;
            this.dims = new short[DIMS][capacity];
            this.labels = new byte[capacity];
        }

        void add(short[] v, byte label) {
            if (n == labels.length) {
                if (!grow) throw new IllegalStateException("capacity exceeded; adjust expected reference count");
                int next = labels.length * 2;
                labels = Arrays.copyOf(labels, next);
                for (int d = 0; d < DIMS; d++) dims[d] = Arrays.copyOf(dims[d], next);
            }
            for (int d = 0; d < DIMS; d++) dims[d][n] = v[d];
            labels[n] = label;
            n++;
        }

        ReferenceIndex build() {
            if (grow) {
                labels = Arrays.copyOf(labels, n);
                for (int d = 0; d < DIMS; d++) dims[d] = Arrays.copyOf(dims[d], n);
            }
            return new ReferenceIndex(n, dims, labels);
        }
    }
}
