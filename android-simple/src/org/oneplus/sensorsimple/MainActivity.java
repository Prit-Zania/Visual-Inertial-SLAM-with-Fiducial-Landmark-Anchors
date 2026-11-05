package org.oneplus.sensorsimple;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.ImageFormat;
import android.graphics.YuvImage;
import android.hardware.Camera;
import android.hardware.ConsumerIrManager;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.hardware.TriggerEvent;
import android.hardware.TriggerEventListener;
import android.os.Build;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Base64;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.content.res.Configuration;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.graphics.drawable.StateListDrawable;
import android.text.InputType;
import android.view.Gravity;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedWriter;
import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.nio.charset.Charset;
import java.util.List;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.TimeUnit;

/** Lightweight project recorder: all exposed sensors + camera + optional Wi-Fi stream. */
public final class MainActivity extends Activity implements SensorEventListener, SurfaceHolder.Callback, Camera.PreviewCallback {
    private static final Charset UTF8 = Charset.forName("UTF-8");
    private final ArrayBlockingQueue<JSONObject> records = new ArrayBlockingQueue<JSONObject>(2048);
    private final Map<Sensor, Integer> sensorKeys = new HashMap<Sensor, Integer>();
    private final TriggerEventListener triggerListener = new TriggerEventListener() {
        @Override public void onTrigger(TriggerEvent event) {
            emitSensor(event.sensor, event.timestamp, SensorManager.SENSOR_STATUS_ACCURACY_HIGH, event.values);
            if (recording) sensorManager.requestTriggerSensor(this, event.sensor);
        }
    };
    private SensorManager sensorManager;
    private Camera camera;
    private SurfaceView preview;
    private EditText hostField;
    private EditText portField;
    private EditText tokenField;
    private TextView status;
    private TextView statusBadge;
    private TextView recBadge;
    private Button startButton;
    private Button stopButton;
    private volatile boolean recording;
    private volatile long lastFrameNs;
    private Thread writerThread;
    private File outputFile;

    private int dp(float value) {
        return (int)(getResources().getDisplayMetrics().density * value + 0.5f);
    }

    private GradientDrawable createCardDrawable(int bgColor, int borderColor, float cornerRadiusDp, int strokeWidthDp) {
        GradientDrawable gd = new GradientDrawable();
        gd.setColor(bgColor);
        gd.setCornerRadius(dp(cornerRadiusDp));
        if (strokeWidthDp > 0) {
            gd.setStroke(dp(strokeWidthDp), borderColor);
        }
        return gd;
    }

    private StateListDrawable createButtonDrawable(int normalColor, int pressedColor, int disabledColor, float cornerRadiusDp) {
        StateListDrawable sld = new StateListDrawable();
        sld.addState(new int[]{android.R.attr.state_pressed}, createCardDrawable(pressedColor, 0, cornerRadiusDp, 0));
        sld.addState(new int[]{-android.R.attr.state_enabled}, createCardDrawable(disabledColor, 0, cornerRadiusDp, 0));
        sld.addState(new int[]{}, createCardDrawable(normalColor, 0, cornerRadiusDp, 0));
        return sld;
    }

    private EditText createStyledInput(String hint, String initialText, int inputType) {
        final EditText et = new EditText(this);
        et.setHint(hint);
        if (initialText != null && initialText.length() > 0) {
            et.setText(initialText);
        }
        et.setSingleLine(true);
        et.setInputType(inputType);
        et.setTextSize(13);
        et.setTextColor(0xFFF8FAFC);
        et.setHintTextColor(0xFF64748B);
        et.setPadding(dp(12), dp(10), dp(12), dp(10));

        final GradientDrawable normalBg = createCardDrawable(0xFF0F172A, 0xFF334155, 8, 1);
        final GradientDrawable focusedBg = createCardDrawable(0xFF0F172A, 0xFF00F2FE, 8, 1);
        et.setBackground(normalBg);
        et.setOnFocusChangeListener(new View.OnFocusChangeListener() {
            @Override public void onFocusChange(View v, boolean hasFocus) {
                et.setBackground(hasFocus ? focusedBg : normalBg);
            }
        });
        return et;
    }

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            getWindow().setStatusBarColor(0xFF070B14);
            getWindow().setNavigationBarColor(0xFF070B14);
        }
        sensorManager = (SensorManager)getSystemService(Context.SENSOR_SERVICE);

        // Root View: Deep AMOLED Dark
        LinearLayout root = new LinearLayout(this);
        root.setBackgroundColor(0xFF070B14);
        int orientation = getResources().getConfiguration().orientation;
        boolean isPortrait = (orientation == Configuration.ORIENTATION_PORTRAIT);
        root.setOrientation(isPortrait ? LinearLayout.VERTICAL : LinearLayout.HORIZONTAL);
        root.setPadding(dp(10), dp(8), dp(10), dp(8));

        // LEFT COLUMN: Scrollable Controls & Telemetry Panel
        ScrollView leftScrollView = new ScrollView(this);
        leftScrollView.setFillViewport(true);
        LinearLayout.LayoutParams leftParams = isPortrait
            ? new LinearLayout.LayoutParams(-1, 0, 1.0f)
            : new LinearLayout.LayoutParams(0, -1, 1.05f);
        if (!isPortrait) {
            leftParams.setMargins(0, 0, dp(10), 0);
        } else {
            leftParams.setMargins(0, 0, 0, dp(10));
        }
        leftScrollView.setLayoutParams(leftParams);

        LinearLayout controls = new LinearLayout(this);
        controls.setOrientation(LinearLayout.VERTICAL);
        controls.setPadding(0, 0, 0, dp(8));

        // Header: OnePlus Nord CE Brand Badge + Status Pill
        LinearLayout headerRow = new LinearLayout(this);
        headerRow.setOrientation(LinearLayout.HORIZONTAL);
        headerRow.setGravity(Gravity.CENTER_VERTICAL);

        TextView brandBadge = new TextView(this);
        brandBadge.setText("ONEPLUS NORD CE");
        brandBadge.setTextSize(10);
        brandBadge.setTypeface(Typeface.create("sans-serif-medium", Typeface.BOLD));
        brandBadge.setTextColor(0xFFFF4D6D);
        brandBadge.setBackground(createCardDrawable(0x28EB0029, 0xFFEB0029, 4, 1));
        brandBadge.setPadding(dp(8), dp(3), dp(8), dp(3));
        headerRow.addView(brandBadge);

        View headerSpacer = new View(this);
        headerRow.addView(headerSpacer, new LinearLayout.LayoutParams(0, 1, 1.0f));

        statusBadge = new TextView(this);
        statusBadge.setText("● READY");
        statusBadge.setTextSize(10);
        statusBadge.setTypeface(Typeface.create("sans-serif-medium", Typeface.BOLD));
        statusBadge.setTextColor(0xFF10B981);
        statusBadge.setBackground(createCardDrawable(0x2010B981, 0xFF10B981, 4, 1));
        statusBadge.setPadding(dp(8), dp(3), dp(8), dp(3));
        headerRow.addView(statusBadge);
        controls.addView(headerRow);

        // App Title & Subtitle
        TextView title = new TextView(this);
        title.setText("Sensor & SLAM Studio");
        title.setTextSize(18);
        title.setTextColor(0xFFF8FAFC);
        title.setTypeface(Typeface.create("sans-serif", Typeface.BOLD));
        LinearLayout.LayoutParams titleParams = new LinearLayout.LayoutParams(-2, -2);
        titleParams.setMargins(0, dp(4), 0, 0);
        controls.addView(title, titleParams);

        TextView subtitle = new TextView(this);
        subtitle.setText("Visual-Inertial Odometry • IMU • High-Rate Stream");
        subtitle.setTextSize(11);
        subtitle.setTextColor(0xFF94A3B8);
        LinearLayout.LayoutParams subParams = new LinearLayout.LayoutParams(-2, -2);
        subParams.setMargins(0, dp(1), 0, dp(8));
        controls.addView(subtitle, subParams);

        // CARD 1: Stream Connection Settings
        LinearLayout connCard = new LinearLayout(this);
        connCard.setOrientation(LinearLayout.VERTICAL);
        connCard.setBackground(createCardDrawable(0xFF111827, 0xFF1F2937, 10, 1));
        connCard.setPadding(dp(12), dp(10), dp(12), dp(10));
        LinearLayout.LayoutParams connCardParams = new LinearLayout.LayoutParams(-1, -2);
        connCardParams.setMargins(0, 0, 0, dp(8));
        connCard.setLayoutParams(connCardParams);

        TextView connHeader = new TextView(this);
        connHeader.setText("STREAM DESTINATION (TCP)");
        connHeader.setTextSize(10);
        connHeader.setTypeface(Typeface.create("sans-serif-medium", Typeface.BOLD));
        connHeader.setTextColor(0xFF38BDF8);
        connCard.addView(connHeader);

        TextView ipLabel = new TextView(this);
        ipLabel.setText("Laptop Wi-Fi IPv4");
        ipLabel.setTextSize(11);
        ipLabel.setTextColor(0xFF94A3B8);
        LinearLayout.LayoutParams ipLabelParams = new LinearLayout.LayoutParams(-2, -2);
        ipLabelParams.setMargins(0, dp(6), 0, dp(3));
        connCard.addView(ipLabel, ipLabelParams);

        hostField = createStyledInput("192.168.x.x (Leave blank for local storage)", "", InputType.TYPE_CLASS_TEXT);
        connCard.addView(hostField, new LinearLayout.LayoutParams(-1, -2));

        LinearLayout portTokenRow = new LinearLayout(this);
        portTokenRow.setOrientation(LinearLayout.HORIZONTAL);
        LinearLayout.LayoutParams ptrParams = new LinearLayout.LayoutParams(-1, -2);
        ptrParams.setMargins(0, dp(6), 0, 0);
        portTokenRow.setLayoutParams(ptrParams);

        LinearLayout portCol = new LinearLayout(this);
        portCol.setOrientation(LinearLayout.VERTICAL);
        TextView portLabel = new TextView(this);
        portLabel.setText("Port");
        portLabel.setTextSize(11);
        portLabel.setTextColor(0xFF94A3B8);
        portCol.addView(portLabel);
        portField = createStyledInput("8765", "8765", InputType.TYPE_CLASS_NUMBER);
        LinearLayout.LayoutParams portInputParams = new LinearLayout.LayoutParams(-1, -2);
        portInputParams.setMargins(0, dp(3), 0, 0);
        portCol.addView(portField, portInputParams);
        LinearLayout.LayoutParams portColParams = new LinearLayout.LayoutParams(0, -2, 1.0f);
        portColParams.setMargins(0, 0, dp(6), 0);
        portTokenRow.addView(portCol, portColParams);

        LinearLayout tokenCol = new LinearLayout(this);
        tokenCol.setOrientation(LinearLayout.VERTICAL);
        TextView tokenLabel = new TextView(this);
        tokenLabel.setText("Pairing Token");
        tokenLabel.setTextSize(11);
        tokenLabel.setTextColor(0xFF94A3B8);
        tokenCol.addView(tokenLabel);
        tokenField = createStyledInput("Receiver Token", "", InputType.TYPE_CLASS_TEXT);
        LinearLayout.LayoutParams tokenInputParams = new LinearLayout.LayoutParams(-1, -2);
        tokenInputParams.setMargins(0, dp(3), 0, 0);
        tokenCol.addView(tokenField, tokenInputParams);
        LinearLayout.LayoutParams tokenColParams = new LinearLayout.LayoutParams(0, -2, 1.5f);
        tokenColParams.setMargins(dp(6), 0, 0, 0);
        portTokenRow.addView(tokenCol, tokenColParams);

        connCard.addView(portTokenRow);
        controls.addView(connCard);

        // CARD 2: Action Buttons
        LinearLayout btnRow = new LinearLayout(this);
        btnRow.setOrientation(LinearLayout.HORIZONTAL);
        LinearLayout.LayoutParams btnRowParams = new LinearLayout.LayoutParams(-1, -2);
        btnRowParams.setMargins(0, 0, 0, dp(8));
        btnRow.setLayoutParams(btnRowParams);

        startButton = new Button(this);
        startButton.setText("▶  START");
        startButton.setTextSize(13);
        startButton.setTypeface(Typeface.create("sans-serif", Typeface.BOLD));
        startButton.setTextColor(0xFFFFFFFF);
        startButton.setBackground(createButtonDrawable(0xFFEB0029, 0xFFB91C1C, 0xFF1F2937, 8));
        LinearLayout.LayoutParams startParams = new LinearLayout.LayoutParams(0, dp(44), 1.2f);
        startParams.setMargins(0, 0, dp(4), 0);
        btnRow.addView(startButton, startParams);

        stopButton = new Button(this);
        stopButton.setText("■  STOP & SAVE");
        stopButton.setTextSize(13);
        stopButton.setTypeface(Typeface.create("sans-serif", Typeface.BOLD));
        stopButton.setTextColor(0xFFCBD5E1);
        stopButton.setBackground(createButtonDrawable(0xFFDC2626, 0xFF991B1B, 0xFF131A26, 8));
        stopButton.setEnabled(false);
        LinearLayout.LayoutParams stopParams = new LinearLayout.LayoutParams(0, dp(44), 1.0f);
        stopParams.setMargins(dp(4), 0, 0, 0);
        btnRow.addView(stopButton, stopParams);

        controls.addView(btnRow);

        // CARD 3: System Telemetry
        LinearLayout logCard = new LinearLayout(this);
        logCard.setOrientation(LinearLayout.VERTICAL);
        logCard.setBackground(createCardDrawable(0xFF0F172A, 0xFF1E293B, 10, 1));
        logCard.setPadding(dp(12), dp(10), dp(12), dp(10));

        TextView logTitle = new TextView(this);
        logTitle.setText("SYSTEM TELEMETRY");
        logTitle.setTextSize(10);
        logTitle.setTypeface(Typeface.create("sans-serif-medium", Typeface.BOLD));
        logTitle.setTextColor(0xFF64748B);
        logCard.addView(logTitle);

        status = new TextView(this);
        status.setText("Ready. Enter laptop IP or record locally on OnePlus Nord.");
        status.setTextSize(11);
        status.setTextColor(0xFFCBD5E1);
        status.setTypeface(Typeface.MONOSPACE);
        LinearLayout.LayoutParams statusParams = new LinearLayout.LayoutParams(-1, -2);
        statusParams.setMargins(0, dp(4), 0, 0);
        logCard.addView(status, statusParams);

        controls.addView(logCard);
        leftScrollView.addView(controls);
        root.addView(leftScrollView);

        // RIGHT COLUMN: Camera Viewport Card
        LinearLayout camCard = new LinearLayout(this);
        camCard.setOrientation(LinearLayout.VERTICAL);
        camCard.setBackground(createCardDrawable(0xFF070B14, 0xFF1E293B, 10, 1));
        camCard.setPadding(dp(8), dp(8), dp(8), dp(8));
        LinearLayout.LayoutParams camCardParams = isPortrait
            ? new LinearLayout.LayoutParams(-1, 0, 1.0f)
            : new LinearLayout.LayoutParams(0, -1, 1.15f);
        camCard.setLayoutParams(camCardParams);

        LinearLayout camHeader = new LinearLayout(this);
        camHeader.setOrientation(LinearLayout.HORIZONTAL);
        camHeader.setGravity(Gravity.CENTER_VERTICAL);
        camHeader.setPadding(dp(4), 0, dp(4), dp(6));

        TextView camTitle = new TextView(this);
        camTitle.setText("OPTICAL SENSOR VIEWPORT");
        camTitle.setTextSize(10);
        camTitle.setTypeface(Typeface.create("sans-serif-medium", Typeface.BOLD));
        camTitle.setTextColor(0xFF94A3B8);
        camHeader.addView(camTitle);

        View camSpacer = new View(this);
        camHeader.addView(camSpacer, new LinearLayout.LayoutParams(0, 1, 1.0f));

        recBadge = new TextView(this);
        recBadge.setText("● REC LIVE");
        recBadge.setTextSize(9);
        recBadge.setTypeface(Typeface.create("sans-serif-medium", Typeface.BOLD));
        recBadge.setTextColor(0xFFFF4D4D);
        recBadge.setBackground(createCardDrawable(0x35EF4444, 0xFFEF4444, 4, 1));
        recBadge.setPadding(dp(6), dp(2), dp(6), dp(2));
        recBadge.setVisibility(View.GONE);
        LinearLayout.LayoutParams recParams = new LinearLayout.LayoutParams(-2, -2);
        recParams.setMargins(0, 0, dp(6), 0);
        camHeader.addView(recBadge, recParams);

        TextView resBadge = new TextView(this);
        resBadge.setText("640 × 480 NV21");
        resBadge.setTextSize(9);
        resBadge.setTypeface(Typeface.create("sans-serif-medium", Typeface.BOLD));
        resBadge.setTextColor(0xFF00F2FE);
        resBadge.setBackground(createCardDrawable(0x2000F2FE, 0xFF00F2FE, 4, 1));
        resBadge.setPadding(dp(6), dp(2), dp(6), dp(2));
        camHeader.addView(resBadge);

        camCard.addView(camHeader);

        preview = new SurfaceView(this);
        preview.getHolder().addCallback(this);
        camCard.addView(preview, new LinearLayout.LayoutParams(-1, -1));

        root.addView(camCard);
        setContentView(root);

        startButton.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View view) { startCapture(); }
        });
        stopButton.setOnClickListener(new View.OnClickListener() {
            @Override public void onClick(View view) { stopCapture(); }
        });
    }

    private void startCapture() {
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, 10);
            return;
        }
        if (!preview.getHolder().getSurface().isValid()) {
            setStatus("Camera surface is not ready yet.");
            return;
        }
        try {
            final String session = UUID.randomUUID().toString();
            File directory = new File(getExternalFilesDir(null), "sessions/" + session);
            if (!directory.mkdirs() && !directory.isDirectory()) throw new Exception("Cannot create session folder");
            outputFile = new File(directory, "capture.jsonl");
            recording = true;
            writerThread = new Thread(new WriterTask(hostField.getText().toString().trim(), Integer.parseInt(portField.getText().toString()), tokenField.getText().toString().trim()), "capture-writer");
            writerThread.start();

            JSONObject hello = new JSONObject();
            hello.put("kind", "hello");
            hello.put("protocol", "oneplus-msckf-v1");
            hello.put("session_id", session);
            hello.put("device", Build.MANUFACTURER + " " + Build.MODEL);
            hello.put("android", Build.VERSION.RELEASE);
            hello.put("elapsed_realtime_ns", SystemClock.elapsedRealtimeNanos());
            ConsumerIrManager ir = (ConsumerIrManager)getSystemService(Context.CONSUMER_IR_SERVICE);
            hello.put("ir_emitter", ir != null && ir.hasIrEmitter());
            hello.put("ir_range_available", false);
            JSONArray inventory = new JSONArray();
            List<Sensor> sensors = sensorManager.getSensorList(Sensor.TYPE_ALL);
            sensorKeys.clear();
            int sensorKey = 0;
            for (Sensor sensor : sensors) {
                sensorKeys.put(sensor, ++sensorKey);
                JSONObject item = new JSONObject();
                item.put("key", sensorKey);
                item.put("type", sensor.getType());
                item.put("name", sensor.getName());
                item.put("vendor", sensor.getVendor());
                item.put("version", sensor.getVersion());
                item.put("string_type", sensor.getStringType());
                item.put("wake_up", sensor.isWakeUpSensor());
                item.put("reporting_mode", sensor.getReportingMode());
                item.put("min_delay_us", sensor.getMinDelay());
                item.put("max_delay_us", sensor.getMaxDelay());
                item.put("power_ma", sensor.getPower());
                item.put("resolution", sensor.getResolution());
                item.put("maximum_range", sensor.getMaximumRange());
                inventory.put(item);
            }
            hello.put("available_sensor_count", sensors.size());
            hello.put("sensors", inventory);
            enqueue(hello);
            for (Sensor sensor : sensors) {
                boolean registered;
                if (sensor.getReportingMode() == Sensor.REPORTING_MODE_ONE_SHOT) {
                    registered = sensorManager.requestTriggerSensor(triggerListener, sensor);
                } else {
                    int periodUs = (sensor.getType() == Sensor.TYPE_ACCELEROMETER || sensor.getType() == Sensor.TYPE_GYROSCOPE) ? 5000 :
                        ((sensor.getType() == Sensor.TYPE_MAGNETIC_FIELD || sensor.getType() == Sensor.TYPE_ROTATION_VECTOR) ? 20000 : 50000);
                    registered = sensorManager.registerListener(this, sensor, periodUs, 0);
                }
                enqueue(new JSONObject().put("kind", "sensor_registration")
                    .put("sensor_key", sensorKeys.get(sensor)).put("registered", registered)
                    .put("wake_up", sensor.isWakeUpSensor()).put("reporting_mode", sensor.getReportingMode()));
            }
            openCamera();
            startButton.setEnabled(false);
            stopButton.setEnabled(true);
            stopButton.setTextColor(0xFFFFFFFF);
            if (statusBadge != null) {
                statusBadge.setText("● RECORDING");
                statusBadge.setTextColor(0xFFEF4444);
                statusBadge.setBackground(createCardDrawable(0x28EF4444, 0xFFEF4444, 4, 1));
            }
            if (recBadge != null) {
                recBadge.setVisibility(View.VISIBLE);
            }
            setStatus("Recording " + sensors.size() + " exposed sensors and camera. File: " + outputFile.getAbsolutePath());
        } catch (Exception error) {
            stopCapture();
            setStatus("Cannot start: " + error.getMessage());
        }
    }

    @SuppressWarnings("deprecation") private void openCamera() throws Exception {
        int selected = 0;
        int count = Camera.getNumberOfCameras();
        Camera.CameraInfo info = new Camera.CameraInfo();
        for (int index = 0; index < count; index++) {
            Camera.getCameraInfo(index, info);
            if (info.facing == Camera.CameraInfo.CAMERA_FACING_BACK) { selected = index; break; }
        }
        camera = Camera.open(selected);
        Camera.Parameters parameters = camera.getParameters();
        Camera.Size chosen = parameters.getSupportedPreviewSizes().get(0);
        long best = Long.MAX_VALUE;
        for (Camera.Size size : parameters.getSupportedPreviewSizes()) {
            long difference = Math.abs((long)size.width * size.height - 640L * 480L);
            if (difference < best) { best = difference; chosen = size; }
        }
        parameters.setPreviewSize(chosen.width, chosen.height);
        parameters.setPreviewFormat(ImageFormat.NV21);
        List<String> focusModes = parameters.getSupportedFocusModes();
        if (focusModes.contains(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO)) parameters.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO);
        camera.setParameters(parameters);
        camera.setPreviewDisplay(preview.getHolder());
        camera.setPreviewCallback(this);
        camera.startPreview();
        JSONObject cameraInfo = new JSONObject();
        cameraInfo.put("kind", "camera_info");
        cameraInfo.put("camera_id", selected);
        cameraInfo.put("width", chosen.width);
        cameraInfo.put("height", chosen.height);
        cameraInfo.put("format", "NV21_to_JPEG");
        cameraInfo.put("laser_af_range", "not_exposed_by_standard_android_api");
        enqueue(cameraInfo);
    }

    private void stopCapture() {
        if (recording) {
            try { enqueue(new JSONObject().put("kind", "end")); }
            catch (Exception ignored) { }
        }
        recording = false;
        sensorManager.unregisterListener(this);
        for (Sensor sensor : sensorKeys.keySet()) sensorManager.cancelTriggerSensor(triggerListener, sensor);
        if (camera != null) {
            camera.setPreviewCallback(null);
            camera.stopPreview();
            camera.release();
            camera = null;
        }
        startButton.setEnabled(true);
        stopButton.setEnabled(false);
        stopButton.setTextColor(0xFFCBD5E1);
        if (statusBadge != null) {
            statusBadge.setText("● READY");
            statusBadge.setTextColor(0xFF10B981);
            statusBadge.setBackground(createCardDrawable(0x2010B981, 0xFF10B981, 4, 1));
        }
        if (recBadge != null) {
            recBadge.setVisibility(View.GONE);
        }
        if (outputFile != null) setStatus("Saved: " + outputFile.getAbsolutePath());
    }

    private void enqueue(JSONObject object) {
        if (!records.offer(object)) setStatus("Recorder queue full; slow down phone movement.");
    }

    @Override public void onSensorChanged(SensorEvent event) {
        emitSensor(event.sensor, event.timestamp, event.accuracy, event.values);
    }

    private void emitSensor(Sensor sensor, long timestamp, int accuracy, float[] sampleValues) {
        if (!recording) return;
        try {
            JSONObject record = new JSONObject();
            record.put("kind", "sensor");
            record.put("sensor_type", sensor.getType());
            Integer sensorKey = sensorKeys.get(sensor);
            record.put("sensor_key", sensorKey == null ? 0 : sensorKey.intValue());
            record.put("sensor_name", sensor.getName());
            record.put("wake_up", sensor.isWakeUpSensor());
            record.put("timestamp_ns", timestamp);
            record.put("accuracy", accuracy);
            JSONArray values = new JSONArray();
            for (float value : sampleValues) values.put(value);
            record.put("values", values);
            enqueue(record);
        } catch (Exception error) { setStatus("Sensor error: " + error.getMessage()); }
    }

    @Override public void onAccuracyChanged(Sensor sensor, int accuracy) { }

    @SuppressWarnings("deprecation") @Override public void onPreviewFrame(byte[] data, Camera source) {
        if (!recording) return;
        long now = SystemClock.elapsedRealtimeNanos();
        if (now - lastFrameNs < 200000000L) return;
        lastFrameNs = now;
        try {
            Camera.Size size = source.getParameters().getPreviewSize();
            YuvImage image = new YuvImage(data, ImageFormat.NV21, size.width, size.height, null);
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            image.compressToJpeg(new android.graphics.Rect(0, 0, size.width, size.height), 72, bytes);
            JSONObject frame = new JSONObject();
            frame.put("kind", "frame");
            frame.put("timestamp_ns", now);
            frame.put("width", size.width);
            frame.put("height", size.height);
            frame.put("format", "jpeg");
            frame.put("image_b64", Base64.encodeToString(bytes.toByteArray(), Base64.NO_WRAP));
            enqueue(frame);
        } catch (Exception error) { setStatus("Camera frame error: " + error.getMessage()); }
    }

    private void setStatus(final String text) {
        runOnUiThread(new Runnable() { @Override public void run() { status.setText(text); } });
    }

    @Override public void surfaceCreated(SurfaceHolder holder) { }
    @Override public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) { }
    @Override public void surfaceDestroyed(SurfaceHolder holder) { stopCapture(); }
    @Override protected void onPause() { stopCapture(); super.onPause(); }

    @Override public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] results) {
        super.onRequestPermissionsResult(requestCode, permissions, results);
        if (requestCode == 10 && results.length > 0 && results[0] == PackageManager.PERMISSION_GRANTED) startCapture();
        else setStatus("Camera permission is required.");
    }

    private final class WriterTask implements Runnable {
        private final String host;
        private final int port;
        private final String token;
        WriterTask(String host, int port, String token) { this.host = host; this.port = port; this.token = token; }
        @Override public void run() {
            Socket socket = null;
            BufferedWriter network = null;
            BufferedWriter disk = null;
            try {
                disk = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFile), UTF8), 262144);
                if (host.length() > 0) {
                    try {
                        socket = new Socket();
                        socket.connect(new InetSocketAddress(host, port), 4000);
                        network = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream(), UTF8));
                    } catch (Exception networkError) {
                        try { if (socket != null) socket.close(); } catch (Exception ignored) { }
                        socket = null;
                        network = null;
                        setStatus("Laptop connection unavailable; local recording continues.");
                    }
                }
                boolean first = true;
                long sequence = 0;
                while (recording || !records.isEmpty()) {
                    JSONObject record = records.poll(200, TimeUnit.MILLISECONDS);
                    if (record == null) continue;
                    String line;
                    if (first) {
                        line = record.toString();
                        disk.write(line); disk.newLine(); disk.flush();
                        if (network != null) {
                            try {
                                JSONObject authenticatedHello = new JSONObject(line);
                                authenticatedHello.put("token", token);
                                network.write(authenticatedHello.toString()); network.newLine(); network.flush();
                                final BufferedReader feedback = new BufferedReader(new InputStreamReader(socket.getInputStream(), UTF8));
                                socket.setSoTimeout(8000);
                                String reply = feedback.readLine();
                                if (reply == null || !"ready".equals(new JSONObject(reply).optString("kind"))) throw new Exception("Pairing rejected");
                                socket.setSoTimeout(0);
                                new Thread(new Runnable() {
                                    @Override public void run() {
                                        try {
                                            String response;
                                            while ((response = feedback.readLine()) != null) {
                                                JSONObject report = new JSONObject(response);
                                                if ("markers".equals(report.optString("kind"))) {
                                                    setStatus("ArUco IDs: " + report.optJSONArray("ids") + " | frames: " + report.optInt("frames_saved") + " | IMU: " + (report.optBoolean("imu_recent") ? "OK" : "waiting"));
                                                }
                                            }
                                        } catch (Exception ignored) { }
                                    }
                                }, "receiver-feedback").start();
                            } catch (Exception pairingError) {
                                try { network.close(); } catch (Exception ignored) { }
                                try { socket.close(); } catch (Exception ignored) { }
                                network = null;
                                socket = null;
                                setStatus("Pairing failed; local recording continues. Check IP and token.");
                            }
                        }
                        first = false;
                        continue;
                    }
                    record.put("seq", ++sequence);
                    line = record.toString();
                    disk.write(line); disk.newLine();
                    if (network != null) {
                        try { network.write(line); network.newLine(); network.flush(); }
                        catch (Exception networkError) {
                            try { network.close(); } catch (Exception ignored) { }
                            try { if (socket != null) socket.close(); } catch (Exception ignored) { }
                            network = null;
                            socket = null;
                            setStatus("Wi-Fi stream stopped; local recording continues.");
                        }
                    }
                }
                disk.flush(); disk.close();
            } catch (final Exception error) {
                setStatus("Recording error: " + error.getMessage() + ". File: " + outputFile);
            } finally {
                try { if (disk != null) disk.close(); } catch (Exception ignored) { }
                try { if (network != null) network.close(); } catch (Exception ignored) { }
                try { if (socket != null) socket.close(); } catch (Exception ignored) { }
            }
        }
    }
}
