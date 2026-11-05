package org.oneplus.slam;

import org.json.JSONObject;
import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;

/** Disk is authoritative. Network backpressure never blocks sensor callbacks. */
public final class SessionWriter {
    public interface Listener { void status(String value); }
    private final ArrayBlockingQueue<JSONObject> diskQueue = new ArrayBlockingQueue<>(4096);
    private final ArrayBlockingQueue<String> networkQueue = new ArrayBlockingQueue<>(512);
    private final Listener listener;
    private final File output;
    private final String host, token;
    private final int port;
    private final JSONObject hello;
    private volatile boolean accepting = true, writing = true;
    private volatile Socket socket;
    public final AtomicLong diskDrops = new AtomicLong(), networkDrops = new AtomicLong();
    public SessionWriter(File directory, String host, int port, String token, JSONObject hello, Listener listener) {
        this.output = new File(directory, "capture.jsonl"); this.host=host; this.port=port;
        this.token=token; this.hello=hello; this.listener=listener;
        new Thread(this::diskLoop,"session-disk").start();
        if (!host.isEmpty()) new Thread(this::networkLoop,"session-network").start();
    }
    public void offer(JSONObject record) {
        if (accepting && !diskQueue.offer(record)) diskDrops.incrementAndGet();
    }
    public void status(String message) { listener.status(message); }
    public void stop() {
        accepting=false;
        // Bound a blocked network write; disk draining remains independent.
        Thread closer=new Thread(()-> {
            try { Thread.sleep(4000);Socket s=socket;if(s!=null)s.close(); } catch(Exception ignored) { }
        },"network-shutdown");closer.setDaemon(true);closer.start();
    }
    private void diskLoop() {
        long seq=0, flushAt=System.nanoTime();
        try (BufferedWriter out=new BufferedWriter(new OutputStreamWriter(new FileOutputStream(output),StandardCharsets.UTF_8),262144)) {
            out.write(hello.toString()); out.newLine(); out.flush();
            while (accepting || !diskQueue.isEmpty()) {
                JSONObject value=diskQueue.poll(100,TimeUnit.MILLISECONDS);
                if (value==null) continue;
                value.put("seq",++seq);
                String line=value.toString(); out.write(line); out.newLine();
                if (!host.isEmpty() && !networkQueue.offer(line)) networkDrops.incrementAndGet();
                if (System.nanoTime()-flushAt>1_000_000_000L) { out.flush(); flushAt=System.nanoTime(); }
            }
            JSONObject end=new JSONObject().put("kind","end").put("seq",++seq)
                .put("disk_dropped_records",diskDrops.get()).put("network_dropped_records",networkDrops.get());
            out.write(end.toString()); out.newLine(); out.flush();
            if (!host.isEmpty()) networkQueue.offer(end.toString());
            listener.status("Saved: "+output.getAbsolutePath()+"; capture drops="+diskDrops.get()+"; Wi-Fi drops="+networkDrops.get());
        } catch(Exception e) { listener.status("RECORDING ERROR: "+e.getMessage()); }
        finally { accepting=false; writing=false; }
    }
    private void networkLoop() {
        // A disconnect requires a new connection/recording. Local data remains complete.
        try (Socket connected=new Socket()) {
            socket=connected; connected.connect(new InetSocketAddress(host,port),4000);
            connected.setSoTimeout(8000); connected.setTcpNoDelay(true);
            BufferedWriter out=new BufferedWriter(new OutputStreamWriter(connected.getOutputStream(),StandardCharsets.UTF_8));
            BufferedReader in=new BufferedReader(new InputStreamReader(connected.getInputStream(),StandardCharsets.UTF_8));
            JSONObject auth=new JSONObject(hello.toString()).put("token",token);
            out.write(auth.toString()); out.newLine(); out.flush();
            String reply=in.readLine();
            if (reply==null || !"ready".equals(new JSONObject(reply).optString("kind"))) throw new IOException("Pairing rejected");
            connected.setSoTimeout(0);
            Thread feedback=new Thread(()-> {
                try {
                    String line;
                    while((line=in.readLine())!=null) {
                        JSONObject report=new JSONObject(line);
                        if("markers".equals(report.optString("kind"))) {
                            listener.status("Markers: "+report.optJSONArray("ids")+
                                "  |  Frames saved on laptop: "+report.optInt("frames_saved")+
                                "  |  IMU: "+(report.optBoolean("imu_recent")?"receiving":"missing or delayed")+
                                "  |  Network gaps: "+report.optInt("sequence_gaps"));
                        }
                    }
                }
                catch(Exception ignored) { }
            },"laptop-feedback"); feedback.setDaemon(true); feedback.start();
            while (writing || !networkQueue.isEmpty()) {
                String line=networkQueue.poll(200,TimeUnit.MILLISECONDS);
                if(line!=null) { out.write(line); out.newLine(); out.flush(); }
            }
        } catch(Exception e) { listener.status("Wi-Fi unavailable; local recording continues: "+e.getMessage()); }
        finally { socket=null; }
    }
}
