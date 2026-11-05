package org.oneplus.slam;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.os.*;
import android.view.*;
import android.widget.*;
import org.json.JSONObject;
import java.io.File;
import java.util.UUID;

public final class MainActivity extends Activity {
    private SurfaceView preview;
    private EditText host,port,token;
    private TextView status;
    private CheckBox depth;
    private Button start,stop;
    private SessionWriter writer;
    private CameraCapture camera;
    private SensorCapture sensors;
    private HandlerThread captureThread;
    @Override public void onCreate(Bundle state) {
        super.onCreate(state); getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        LinearLayout root=new LinearLayout(this); root.setOrientation(LinearLayout.VERTICAL); root.setPadding(18,12,18,12);
        TextView title=new TextView(this); title.setText("OnePlus Nord CE • Camera + IMU + ArUco SLAM"); title.setTextSize(21); root.addView(title);
        LinearLayout settings=new LinearLayout(this);
        host=entry(settings,"Laptop Wi-Fi IP (blank = local)","",3); port=entry(settings,"Port","8765",1); token=entry(settings,"Pairing token","",2);
        root.addView(settings);
        LinearLayout buttons=new LinearLayout(this); start=new Button(this);start.setText("Start recording");stop=new Button(this);stop.setText("Stop & save");stop.setEnabled(false);
        depth=new CheckBox(this);depth.setText("Try exposed depth sensor");buttons.addView(start);buttons.addView(stop);buttons.addView(depth);root.addView(buttons);
        status=new TextView(this);status.setText("Hold still for the first 2 seconds. Camera/IMU calibration is required before reliable SLAM.");root.addView(status);
        preview=new SurfaceView(this);root.addView(preview,new LinearLayout.LayoutParams(-1,0,1));setContentView(root);
        start.setOnClickListener(v->begin());stop.setOnClickListener(v->finishRecording());
    }
    private EditText entry(LinearLayout row,String hint,String value,int weight) {
        EditText e=new EditText(this);e.setHint(hint);e.setText(value);e.setSingleLine();row.addView(e,new LinearLayout.LayoutParams(0,-2,weight));return e;
    }
    private void show(String value) { runOnUiThread(()->status.setText(value)); }
    private void begin() {
        if(checkSelfPermission(Manifest.permission.CAMERA)!=PackageManager.PERMISSION_GRANTED) { requestPermissions(new String[]{Manifest.permission.CAMERA},10);return; }
        if(!preview.getHolder().getSurface().isValid()) { show("Camera preview is not ready yet");return; }
        try {
            int number=Integer.parseInt(port.getText().toString());if(number<1||number>65535) throw new IllegalArgumentException("Port must be 1–65535");
            String id=UUID.randomUUID().toString();File dir=new File(getExternalFilesDir(null),"sessions/"+id);
            if(!dir.mkdirs()) throw new IllegalStateException("Cannot create session folder");
            JSONObject hello=SensorCapture.inventory(this).put("kind","hello").put("protocol","oneplus-msckf-v1")
                .put("session_id",id).put("device",Build.MANUFACTURER+" "+Build.MODEL).put("android",Build.VERSION.RELEASE)
                .put("elapsed_realtime_ns",SystemClock.elapsedRealtimeNanos()).put("wall_time_ms",System.currentTimeMillis());
            writer=new SessionWriter(dir,host.getText().toString().trim(),number,token.getText().toString(),hello,this::show);
            preview.getHolder().setFixedSize(640,480);
            captureThread=new HandlerThread("camera-sensors");captureThread.start();Handler handler=new Handler(captureThread.getLooper());
            sensors=new SensorCapture(this,writer,handler);sensors.start();
            camera=new CameraCapture(this,writer,handler,preview.getHolder().getSurface(),depth.isChecked());camera.start();
            start.setEnabled(false);stop.setEnabled(true);show("Recording. Hold still for 2 seconds, then move slowly and observe markers. Local copy: "+dir);
        } catch(Exception e) { finishRecording();show("Cannot start: "+e.getMessage()); }
    }
    private void finishRecording() {
        if(camera!=null) {camera.stop();camera=null;}
        if(sensors!=null) {sensors.stop();sensors=null;}
        if(writer!=null) {writer.stop();writer=null;}
        if(captureThread!=null) {captureThread.quitSafely();captureThread=null;}
        start.setEnabled(true);stop.setEnabled(false);
    }
    @Override public void onRequestPermissionsResult(int code,String[] permissions,int[] results) {
        super.onRequestPermissionsResult(code,permissions,results);
        if(code==10&&results.length>0&&results[0]==PackageManager.PERMISSION_GRANTED) begin();
        else show("Camera permission is needed to record images");
    }
    @Override protected void onPause() { finishRecording();super.onPause(); }
}
