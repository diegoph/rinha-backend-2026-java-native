package rinha.response;

import java.nio.charset.StandardCharsets;

public final class Responses {
    private Responses() {}

    public static final byte[] READY = raw(200, "text/plain", "OK");
    public static final byte[] BAD_REQUEST = raw(400, "text/plain", "bad request");
    public static final byte[] NOT_FOUND = raw(404, "text/plain", "not found");
    public static final byte[] PAYLOAD_TOO_LARGE = raw(413, "text/plain", "payload too large");

    public static final byte[][] FRAUD_RESPONSES = new byte[][]{
            fraud(true, "0.0"),
            fraud(true, "0.2"),
            fraud(true, "0.4"),
            fraud(false, "0.6"),
            fraud(false, "0.8"),
            fraud(false, "1.0")
    };

    private static byte[] fraud(boolean approved, String score) {
        return raw(200, "application/json", "{\"approved\":" + approved + ",\"fraud_score\":" + score + "}");
    }

    private static byte[] raw(int status, String contentType, String body) {
        String reason = switch (status) {
            case 200 -> "OK";
            case 400 -> "Bad Request";
            case 404 -> "Not Found";
            case 413 -> "Payload Too Large";
            default -> "Error";
        };
        byte[] b = body.getBytes(StandardCharsets.US_ASCII);
        String head = "HTTP/1.1 " + status + " " + reason + "\r\n" +
                "Content-Type: " + contentType + "\r\n" +
                "Content-Length: " + b.length + "\r\n" +
                "Connection: keep-alive\r\n" +
                "\r\n";
        byte[] h = head.getBytes(StandardCharsets.US_ASCII);
        byte[] out = new byte[h.length + b.length];
        System.arraycopy(h, 0, out, 0, h.length);
        System.arraycopy(b, 0, out, h.length, b.length);
        return out;
    }
}
