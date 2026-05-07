package rinha.vector;

import java.nio.charset.StandardCharsets;

public final class Vectorizer {
    private static final int SCALE = 10_000;

    private static final byte[] K_TRANSACTION = ascii("\"transaction\"");
    private static final byte[] K_CUSTOMER = ascii("\"customer\"");
    private static final byte[] K_MERCHANT = ascii("\"merchant\"");
    private static final byte[] K_TERMINAL = ascii("\"terminal\"");
    private static final byte[] K_LAST_TRANSACTION = ascii("\"last_transaction\"");
    private static final byte[] K_AMOUNT = ascii("\"amount\"");
    private static final byte[] K_INSTALLMENTS = ascii("\"installments\"");
    private static final byte[] K_REQUESTED_AT = ascii("\"requested_at\"");
    private static final byte[] K_AVG_AMOUNT = ascii("\"avg_amount\"");
    private static final byte[] K_TX_COUNT_24H = ascii("\"tx_count_24h\"");
    private static final byte[] K_ID = ascii("\"id\"");
    private static final byte[] K_MCC = ascii("\"mcc\"");
    private static final byte[] K_IS_ONLINE = ascii("\"is_online\"");
    private static final byte[] K_CARD_PRESENT = ascii("\"card_present\"");
    private static final byte[] K_KM_FROM_HOME = ascii("\"km_from_home\"");
    private static final byte[] K_KNOWN_MERCHANTS = ascii("\"known_merchants\"");
    private static final byte[] K_TIMESTAMP = ascii("\"timestamp\"");
    private static final byte[] K_KM_FROM_CURRENT = ascii("\"km_from_current\"");


    public void vectorizeToInt16(byte[] body, int len, short[] out) {
        int tx = FastJson.objectStart(body, len, K_TRANSACTION, 0);
        int customer = FastJson.objectStart(body, len, K_CUSTOMER, 0);
        int merchant = FastJson.objectStart(body, len, K_MERCHANT, 0);
        int terminal = FastJson.objectStart(body, len, K_TERMINAL, 0);
        int lastKey = FastJson.findKey(body, len, K_LAST_TRANSACTION, 0);

        double amount = FastJson.numberAfterKey(body, len, K_AMOUNT, tx);
        int installments = FastJson.intAfterKey(body, len, K_INSTALLMENTS, tx);
        long requestedAt = FastJson.stringRangeAfterKey(body, len, K_REQUESTED_AT, tx);
        int requestedAtStart = (int) (requestedAt >>> 32);

        double customerAvg = FastJson.numberAfterKey(body, len, K_AVG_AMOUNT, customer);
        int txCount24h = FastJson.intAfterKey(body, len, K_TX_COUNT_24H, customer);

        long merchantId = FastJson.stringRangeAfterKey(body, len, K_ID, merchant);
        long mcc = FastJson.stringRangeAfterKey(body, len, K_MCC, merchant);
        double merchantAvg = FastJson.numberAfterKey(body, len, K_AVG_AMOUNT, merchant);

        boolean isOnline = FastJson.boolAfterKey(body, len, K_IS_ONLINE, terminal);
        boolean cardPresent = FastJson.boolAfterKey(body, len, K_CARD_PRESENT, terminal);
        double kmFromHome = FastJson.numberAfterKey(body, len, K_KM_FROM_HOME, terminal);

        boolean unknownMerchant = !FastJson.stringInArray(body, len, K_KNOWN_MERCHANTS, customer, merchantId);

        out[0] = quantize(clamp(amount / 10_000.0));
        out[1] = quantize(clamp(installments / 12.0));
        out[2] = quantize(customerAvg <= 0 ? 1.0 : clamp((amount / customerAvg) / 10.0));
        out[3] = quantize(parseHour(body, requestedAtStart) / 23.0);
        out[4] = quantize(dayOfWeekMonday0(body, requestedAtStart) / 6.0);

        if (lastKey < 0 || FastJson.isNullAfterKey(body, len, K_LAST_TRANSACTION, 0)) {
            out[5] = -SCALE;
            out[6] = -SCALE;
        } else {
            long lastTs = FastJson.stringRangeAfterKey(body, len, K_TIMESTAMP, lastKey);
            long reqMin = epochMinute(body, requestedAtStart);
            long lastMin = epochMinute(body, (int) (lastTs >>> 32));
            out[5] = quantize(clamp(Math.max(0, reqMin - lastMin) / 1440.0));
            out[6] = quantize(clamp(FastJson.numberAfterKey(body, len, K_KM_FROM_CURRENT, lastKey) / 1000.0));
        }

        out[7] = quantize(clamp(kmFromHome / 1000.0));
        out[8] = quantize(clamp(txCount24h / 20.0));
        out[9] = isOnline ? (short) SCALE : 0;
        out[10] = cardPresent ? (short) SCALE : 0;
        out[11] = unknownMerchant ? (short) SCALE : 0;
        out[12] = quantize(mccRisk(body, (int) (mcc >>> 32), (int) mcc));
        out[13] = quantize(clamp(merchantAvg / 10_000.0));
    }

    private static byte[] ascii(String s) {
        return s.getBytes(StandardCharsets.US_ASCII);
    }

    private static short quantize(double v) {
        return (short) Math.round(v * SCALE);
    }

    private static double clamp(double x) {
        if (x < 0.0) return 0.0;
        if (x > 1.0) return 1.0;
        return x;
    }

    private static int parseHour(byte[] b, int s) {
        return two(b, s + 11);
    }

    private static int dayOfWeekMonday0(byte[] b, int s) {
        int y = four(b, s);
        int m = two(b, s + 5);
        int d = two(b, s + 8);
        long days = daysFromCivil(y, m, d);

        return (int) Math.floorMod(days + 3, 7);
    }

    private static long epochMinute(byte[] b, int s) {
        int y = four(b, s);
        int m = two(b, s + 5);
        int d = two(b, s + 8);
        int hh = two(b, s + 11);
        int mm = two(b, s + 14);
        return daysFromCivil(y, m, d) * 1440L + hh * 60L + mm;
    }


    private static long daysFromCivil(int y, int m, int d) {
        y -= m <= 2 ? 1 : 0;
        long era = Math.floorDiv(y, 400);
        int yoe = (int) (y - era * 400);
        int mp = m + (m > 2 ? -3 : 9);
        int doy = (153 * mp + 2) / 5 + d - 1;
        int doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        return era * 146097L + doe - 719468L;
    }

    private static int two(byte[] b, int p) {
        return (b[p] - '0') * 10 + (b[p + 1] - '0');
    }

    private static int four(byte[] b, int p) {
        return (b[p] - '0') * 1000 + (b[p + 1] - '0') * 100 + (b[p + 2] - '0') * 10 + (b[p + 3] - '0');
    }


    private static double mccRisk(byte[] b, int s, int e) {
        int len = e - s;
        if (len != 4) return 0.5;
        int code = (b[s] - '0') * 1000 + (b[s + 1] - '0') * 100 + (b[s + 2] - '0') * 10 + (b[s + 3] - '0');
        return switch (code) {
            case 5411 -> 0.15;
            case 5812 -> 0.30;
            case 5912 -> 0.20;
            case 5944 -> 0.45;
            case 7801 -> 0.80;
            case 7802 -> 0.75;
            case 7995 -> 0.85;
            case 4511 -> 0.35;
            case 5311 -> 0.25;
            case 5999 -> 0.50;
            default -> 0.50;
        };
    }
}
