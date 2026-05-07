package rinha.http;

import rinha.knn.FraudIndex;
import rinha.knn.IvfIndex;
import rinha.response.Responses;
import rinha.vector.Vectorizer;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicLong;

public final class FastHttpServer {
    private static final byte[] GET_READY = "GET /ready ".getBytes(java.nio.charset.StandardCharsets.US_ASCII);
    private static final byte[] POST_FRAUD = "POST /fraud-score ".getBytes(java.nio.charset.StandardCharsets.US_ASCII);
    private static final byte[] CONTENT_LENGTH = "content-length:".getBytes(java.nio.charset.StandardCharsets.US_ASCII);

    private final int port;
    private final Vectorizer vectorizer;
    private final FraudIndex index;


    private final Semaphore knnSlots = new Semaphore(Integer.getInteger("knn.slots", 1));
    private final ThreadLocal<short[]> queryBuffer = ThreadLocal.withInitial(() -> new short[14]);
    private final boolean profile = Boolean.getBoolean("server.profile");
    private final Metrics metrics = new Metrics();

    public FastHttpServer(int port, Vectorizer vectorizer, FraudIndex index) {
        this.port = port;
        this.vectorizer = vectorizer;
        this.index = index;
    }

    public void start() throws IOException {
        ServerSocket server = new ServerSocket();
        server.setReuseAddress(true);
        server.bind(new InetSocketAddress("0.0.0.0", port), Integer.getInteger("server.backlog", 8192));
        if (profile) startMetricsLogger();

        while (true) {
            Socket socket = server.accept();
            socket.setTcpNoDelay(true);
            socket.setReceiveBufferSize(64 * 1024);
            socket.setSendBufferSize(64 * 1024);
            Thread.startVirtualThread(() -> handleConnection(socket));
        }
    }

    private void handleConnection(Socket socket) {
        try (socket; InputStream in = new BufferedInputStream(socket.getInputStream(), Integer.getInteger("server.inbuf", 4096)); OutputStream out = new BufferedOutputStream(socket.getOutputStream(), Integer.getInteger("server.outbuf", 4096))) {
            handleStreams(in, out);
        } catch (Throwable ignored) {

        }
    }

    public void handleStreams(InputStream in, OutputStream out) throws IOException {
        byte[] header = new byte[Integer.getInteger("server.header.max", 2048)];
        byte[] body = new byte[Integer.getInteger("server.body.max", 4096)];

        while (true) {
            int headerLen = readHeaders(in, header);
            if (headerLen <= 0) return;

            boolean isGetReady = startsWith(header, headerLen, GET_READY);
            boolean isPostFraud = startsWith(header, headerLen, POST_FRAUD);

            int contentLength = contentLength(header, headerLen);
            if (contentLength > body.length) {
                out.write(Responses.PAYLOAD_TOO_LARGE);
                out.flush();
                drain(in, contentLength);
                continue;
            }

            if (contentLength > 0) {
                readFully(in, body, contentLength);
            }

            if (isGetReady) {
                out.write(Responses.READY);
            } else if (isPostFraud) {
                int frauds;
                long t0 = profile ? System.nanoTime() : 0L;
                knnSlots.acquireUninterruptibly();
                try {
                    short[] q = queryBuffer.get();
                    vectorizer.vectorizeToInt16(body, contentLength, q);
                    frauds = index.searchFraudCount(q);
                    if (profile) recordSuccess(t0);
                } catch (Throwable t) {
                    if (profile) metrics.bad.incrementAndGet();
                    out.write(Responses.BAD_REQUEST);
                    out.flush();
                    continue;
                } finally {
                    knnSlots.release();
                }
                out.write(Responses.FRAUD_RESPONSES[frauds]);
            } else {
                out.write(Responses.NOT_FOUND);
            }
            out.flush();
        }
    }

    private void recordSuccess(long t0) {
        long ns = System.nanoTime() - t0;
        metrics.ok.incrementAndGet();
        metrics.ns.addAndGet(ns);
        updateMax(metrics.maxNs, ns);
        long us = ns / 1_000L;
        if (us > 2_000) metrics.gt2ms.incrementAndGet();
        if (us > 10_000) metrics.gt10ms.incrementAndGet();
        if (us > 50_000) metrics.gt50ms.incrementAndGet();
        if (us > 100_000) metrics.gt100ms.incrementAndGet();
        if (index instanceof IvfIndex ivf) {
            IvfIndex.SearchStats st = ivf.lastStats();
            metrics.scanVec.addAndGet(st.scannedVectors);
            metrics.repairVec.addAndGet(st.repairVectors);
            metrics.bbox.addAndGet(st.bboxChecks);
            updateMax(metrics.maxScanVec, st.scannedVectors);
            updateMax(metrics.maxRepairVec, st.repairVectors);
            updateMax(metrics.maxBbox, st.bboxChecks);
        }
    }

    private void startMetricsLogger() {
        Thread.startVirtualThread(() -> {
            long lastOk = 0, lastBad = 0, lastNs = 0, lastScan = 0, lastRepair = 0, lastBbox = 0;
            long lastGt2 = 0, lastGt10 = 0, lastGt50 = 0, lastGt100 = 0;
            while (true) {
                try { Thread.sleep(5_000); } catch (InterruptedException ignored) { return; }
                long ok = metrics.ok.get(), bad = metrics.bad.get(), ns = metrics.ns.get();
                long scan = metrics.scanVec.get(), repair = metrics.repairVec.get(), bbox = metrics.bbox.get();
                long gt2 = metrics.gt2ms.get(), gt10 = metrics.gt10ms.get(), gt50 = metrics.gt50ms.get(), gt100 = metrics.gt100ms.get();
                long dOk = ok - lastOk, dBad = bad - lastBad;
                long dNs = ns - lastNs, dScan = scan - lastScan, dRepair = repair - lastRepair, dBbox = bbox - lastBbox;
                long dGt2 = gt2 - lastGt2, dGt10 = gt10 - lastGt10, dGt50 = gt50 - lastGt50, dGt100 = gt100 - lastGt100;
                if (dOk > 0 || dBad > 0) {
                }
                lastOk = ok; lastBad = bad; lastNs = ns; lastScan = scan; lastRepair = repair; lastBbox = bbox;
                lastGt2 = gt2; lastGt10 = gt10; lastGt50 = gt50; lastGt100 = gt100;
            }
        });
    }

    private static void updateMax(AtomicLong max, long v) {
        long cur;
        do {
            cur = max.get();
            if (v <= cur) return;
        } while (!max.compareAndSet(cur, v));
    }

    private static final class Metrics {
        final AtomicLong ok = new AtomicLong();
        final AtomicLong bad = new AtomicLong();
        final AtomicLong ns = new AtomicLong();
        final AtomicLong maxNs = new AtomicLong();
        final AtomicLong gt2ms = new AtomicLong();
        final AtomicLong gt10ms = new AtomicLong();
        final AtomicLong gt50ms = new AtomicLong();
        final AtomicLong gt100ms = new AtomicLong();
        final AtomicLong scanVec = new AtomicLong();
        final AtomicLong repairVec = new AtomicLong();
        final AtomicLong bbox = new AtomicLong();
        final AtomicLong maxScanVec = new AtomicLong();
        final AtomicLong maxRepairVec = new AtomicLong();
        final AtomicLong maxBbox = new AtomicLong();
    }

    private static int readHeaders(InputStream in, byte[] buf) throws IOException {
        int n = 0;
        int state = 0;
        while (n < buf.length) {
            int b = in.read();
            if (b < 0) return n == 0 ? -1 : n;
            buf[n++] = (byte) b;


            state = switch (state) {
                case 0 -> b == '\r' ? 1 : 0;
                case 1 -> b == '\n' ? 2 : 0;
                case 2 -> b == '\r' ? 3 : 0;
                case 3 -> b == '\n' ? 4 : 0;
                default -> state;
            };
            if (state == 4) return n;
        }
        return -1;
    }

    private static void readFully(InputStream in, byte[] buf, int len) throws IOException {
        int off = 0;
        while (off < len) {
            int r = in.read(buf, off, len - off);
            if (r < 0) throw new IOException("unexpected EOF");
            off += r;
        }
    }

    private static void drain(InputStream in, int len) throws IOException {
        byte[] tmp = new byte[1024];
        int left = len;
        while (left > 0) {
            int r = in.read(tmp, 0, Math.min(tmp.length, left));
            if (r < 0) return;
            left -= r;
        }
    }

    private static boolean startsWith(byte[] a, int len, byte[] s) {
        if (len < s.length) return false;
        for (int i = 0; i < s.length; i++) {
            if (a[i] != s[i]) return false;
        }
        return true;
    }

    private static int contentLength(byte[] header, int len) {
        byte[] key = CONTENT_LENGTH;
        for (int i = 0; i <= len - key.length; i++) {
            int j = 0;
            while (j < key.length) {
                byte hb = header[i + j];
                if (hb >= 'A' && hb <= 'Z') hb = (byte) (hb + 32);
                if (hb != key[j]) break;
                j++;
            }
            if (j == key.length) {
                int p = i + key.length;
                while (p < len && (header[p] == ' ' || header[p] == '\t')) p++;
                int v = 0;
                while (p < len && header[p] >= '0' && header[p] <= '9') {
                    v = v * 10 + (header[p++] - '0');
                }
                return v;
            }
        }
        return 0;
    }
}
