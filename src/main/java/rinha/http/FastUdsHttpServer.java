package rinha.http;

import rinha.knn.FraudIndex;
import rinha.vector.Vectorizer;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.StandardProtocolFamily;
import java.net.UnixDomainSocketAddress;
import java.nio.channels.Channels;
import java.nio.channels.ServerSocketChannel;
import java.nio.channels.SocketChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermissions;

public final class FastUdsHttpServer {
    private final Path socketPath;
    private final FastHttpServer delegate;

    public FastUdsHttpServer(Path socketPath, Vectorizer vectorizer, FraudIndex index) {
        this.socketPath = socketPath;
        this.delegate = new FastHttpServer(-1, vectorizer, index);
    }

    public void start() throws Exception {
        Path parent = socketPath.getParent();
        Files.createDirectories(parent);
        chmod777(parent);
        Files.deleteIfExists(socketPath);

        ServerSocketChannel server = ServerSocketChannel.open(StandardProtocolFamily.UNIX);
        server.bind(UnixDomainSocketAddress.of(socketPath), Integer.getInteger("server.backlog", 8192));
        chmod777(socketPath);

        while (true) {
            SocketChannel ch = server.accept();
            Thread.startVirtualThread(() -> handleConnection(ch));
        }
    }

    private static void chmod777(Path path) {
        try {
            Files.setPosixFilePermissions(path, PosixFilePermissions.fromString("rwxrwxrwx"));
        } catch (Exception ignored) {

        }
    }

    private void handleConnection(SocketChannel ch) {
        try (ch; InputStream in = new BufferedInputStream(Channels.newInputStream(ch), Integer.getInteger("server.inbuf", 4096)); OutputStream out = new BufferedOutputStream(Channels.newOutputStream(ch), Integer.getInteger("server.outbuf", 4096))) {
            delegate.handleStreams(in, out);
        } catch (Throwable ignored) {

        }
    }
}
