package rinha.vector;

final class FastJson {
    private FastJson() {}

    static int findKey(byte[] b, int len, byte[] k, int from) {
        outer:
        for (int i = Math.max(0, from); i <= len - k.length; i++) {

            if (b[i] != k[0]) continue;
            for (int j = 1; j < k.length; j++) {
                if (b[i + j] != k[j]) continue outer;
            }
            return i;
        }
        return -1;
    }

    static int objectStart(byte[] b, int len, byte[] key, int from) {
        int p = findKey(b, len, key, from);
        if (p < 0) return -1;
        p += key.length;
        while (p < len && b[p] != '{') p++;
        return p < len ? p : -1;
    }

    static int arrayStart(byte[] b, int len, byte[] key, int from) {
        int p = findKey(b, len, key, from);
        if (p < 0) return -1;
        p += key.length;
        while (p < len && b[p] != '[') p++;
        return p < len ? p : -1;
    }

    static double numberAfterKey(byte[] b, int len, byte[] key, int from) {
        int p = findKey(b, len, key, from);
        if (p < 0) throw new IllegalArgumentException("missing key");
        p += key.length;
        while (p < len && b[p] != ':') p++;
        if (p >= len) throw new IllegalArgumentException("bad number");
        p++;
        while (p < len && isSpace(b[p])) p++;
        return parseDouble(b, len, p);
    }

    static int intAfterKey(byte[] b, int len, byte[] key, int from) {
        return (int) numberAfterKey(b, len, key, from);
    }

    static boolean boolAfterKey(byte[] b, int len, byte[] key, int from) {
        int p = findKey(b, len, key, from);
        if (p < 0) throw new IllegalArgumentException("missing key");
        p += key.length;
        while (p < len && b[p] != ':') p++;
        p++;
        while (p < len && isSpace(b[p])) p++;
        if (p + 4 <= len && b[p] == 't') return true;
        if (p + 5 <= len && b[p] == 'f') return false;
        throw new IllegalArgumentException("bad bool");
    }

    static long stringRangeAfterKey(byte[] b, int len, byte[] key, int from) {
        int p = findKey(b, len, key, from);
        if (p < 0) throw new IllegalArgumentException("missing key");
        p += key.length;
        while (p < len && b[p] != ':') p++;
        p++;
        while (p < len && b[p] != '"') p++;
        int s = ++p;
        while (p < len && b[p] != '"') p++;
        if (p >= len) throw new IllegalArgumentException("bad string");
        return (((long) s) << 32) | (p & 0xffff_ffffL);
    }

    static boolean isNullAfterKey(byte[] b, int len, byte[] key, int from) {
        int p = findKey(b, len, key, from);
        if (p < 0) return true;
        p += key.length;
        while (p < len && b[p] != ':') p++;
        p++;
        while (p < len && isSpace(b[p])) p++;
        return p < len && b[p] == 'n';
    }

    static boolean stringInArray(byte[] b, int len, byte[] key, int from, long needle) {
        int p = arrayStart(b, len, key, from);
        if (p < 0) return false;
        int needleStart = (int) (needle >>> 32);
        int needleEnd = (int) needle;
        int end = p;
        while (end < len && b[end] != ']') end++;

        for (int i = p; i < end; i++) {
            if (b[i] == '"') {
                int s = i + 1;
                int e = s;
                while (e < end && b[e] != '"') e++;
                if (equals(b, s, e, needleStart, needleEnd)) return true;
                i = e;
            }
        }
        return false;
    }

    private static boolean equals(byte[] b, int s1, int e1, int s2, int e2) {
        int n = e1 - s1;
        if (n != e2 - s2) return false;
        for (int i = 0; i < n; i++) {
            if (b[s1 + i] != b[s2 + i]) return false;
        }
        return true;
    }

    private static boolean isSpace(byte c) {
        return c == ' ' || c == '\n' || c == '\r' || c == '\t';
    }


    private static double parseDouble(byte[] b, int len, int p) {
        boolean neg = false;
        if (p < len && b[p] == '-') {
            neg = true;
            p++;
        }
        long intPart = 0;
        while (p < len && b[p] >= '0' && b[p] <= '9') {
            intPart = intPart * 10 + (b[p++] - '0');
        }
        double v = intPart;
        if (p < len && b[p] == '.') {
            p++;
            double div = 10.0;
            while (p < len && b[p] >= '0' && b[p] <= '9') {
                v += (b[p++] - '0') / div;
                div *= 10.0;
            }
        }
        return neg ? -v : v;
    }
}
