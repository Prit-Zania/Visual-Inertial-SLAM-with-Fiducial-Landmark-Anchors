package org.oneplus.slam;

import android.content.Context;
import android.graphics.*;
import android.hardware.camera2.*;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Handler;
import android.util.*;
import android.view.Surface;
import org.json.*;
import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.util.*;

public final class CameraCapture {
    private final CameraManager manager;
    private final SessionWriter writer;
    private final Handler handler;
    private CameraDevice device;
    private CameraCaptureSession session;
    private ImageReader reader,depthReader;
    private CameraCharacteristics characteristics;
    private volatile boolean stopping;
    private long lastFrame;
    private final Surface preview;
    private final boolean requestDepth;
    public CameraCapture(Context context,SessionWriter writer,Handler handler,Surface preview,boolean depth) {
        manager=(CameraManager)context.getSystemService(Context.CAMERA_SERVICE);
        this.writer=writer; this.handler=handler; this.preview=preview; requestDepth=depth;
    }
    private void event(String kind,String message) {
        writer.status("Camera: "+message);
        try { writer.offer(new JSONObject().put("kind",kind).put("message",message)); } catch(JSONException ignored) { }
    }
    private static JSONArray array(float[] values) throws JSONException {
        JSONArray a=new JSONArray(); if(values!=null) for(float v:values) a.put(v); return a;
    }
    private static boolean contains(int[] values,int target) {
        if(values!=null) for(int v:values) if(v==target) return true; return false;
    }
    @SuppressWarnings("MissingPermission") public void start() throws Exception {
        String selected=null; float best=Float.MAX_VALUE;
        // Prefer a rear camera with a normal field of view over ultrawide / macro cameras.
        for(String id:manager.getCameraIdList()) {
            CameraCharacteristics c=manager.getCameraCharacteristics(id);
            if(!Integer.valueOf(CameraCharacteristics.LENS_FACING_BACK).equals(c.get(CameraCharacteristics.LENS_FACING))) continue;
            float[] focal=c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS);
            SizeF size=c.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE);
            if(focal==null||focal.length==0||size==null) continue;
            float score=Math.abs(focal[0]/size.getWidth()-.75f);
            if(score<best) { best=score; selected=id; }
        }
        if(selected==null) throw new IllegalStateException("No rear camera exposed");
        characteristics=manager.getCameraCharacteristics(selected);
        StreamConfigurationMap map=characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
        Size[] sizes=map.getOutputSizes(ImageFormat.YUV_420_888);
        Size chosen=Arrays.stream(sizes).min(Comparator.comparingLong(s->Math.abs((long)s.getWidth()*s.getHeight()-640L*480))).get();
        reader=ImageReader.newInstance(chosen.getWidth(),chosen.getHeight(),ImageFormat.YUV_420_888,4);
        reader.setOnImageAvailableListener(this::image,handler);
        int[] caps=characteristics.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES);
        JSONObject info=new JSONObject().put("kind","camera_info").put("camera_id",selected)
            .put("width",chosen.getWidth()).put("height",chosen.getHeight())
            .put("timestamp_source",characteristics.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE))
            .put("sensor_orientation",characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION))
            .put("focus_calibration",characteristics.get(CameraCharacteristics.LENS_INFO_FOCUS_DISTANCE_CALIBRATION))
            .put("depth_capability",contains(caps,CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_DEPTH_OUTPUT))
            .put("intrinsics",array(characteristics.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)))
            .put("lens_pose_rotation",array(characteristics.get(CameraCharacteristics.LENS_POSE_ROTATION)))
            .put("lens_pose_translation",array(characteristics.get(CameraCharacteristics.LENS_POSE_TRANSLATION)))
            .put("distortion",array(characteristics.get(CameraCharacteristics.LENS_DISTORTION)))
            .put("raw_laser_range_status","not_exposed_by_standard_api");
        JSONArray vendorCandidates=new JSONArray();
        for(CaptureResult.Key<?> key:characteristics.getAvailableCaptureResultKeys()) {
            String name=key.getName().toLowerCase(java.util.Locale.ROOT);
            if(name.contains("laser")||name.contains("depth")||name.contains("distance")||name.contains("range")) vendorCandidates.put(key.getName());
        }
        info.put("range_related_result_keys",vendorCandidates);
        writer.offer(info);
        if(requestDepth && contains(caps,CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_DEPTH_OUTPUT)) {
            Size[] ds=map.getOutputSizes(ImageFormat.DEPTH16);
            if(ds!=null&&ds.length>0) {
                Size size=Arrays.stream(ds).min(Comparator.comparingLong(s->(long)s.getWidth()*s.getHeight())).get();
                depthReader=ImageReader.newInstance(size.getWidth(),size.getHeight(),ImageFormat.DEPTH16,3);
                depthReader.setOnImageAvailableListener(this::depth,handler);
            }
        }
        manager.openCamera(selected,new CameraDevice.StateCallback() {
            @Override public void onOpened(CameraDevice camera) { if(stopping) {camera.close();return;} device=camera; configure(); }
            @Override public void onDisconnected(CameraDevice camera) { event("camera_error","disconnected"); camera.close(); }
            @Override public void onError(CameraDevice camera,int error) { event("camera_error","error "+error); camera.close(); }
        },handler);
    }
    private void configure() {
        try {
            List<Surface> surfaces=new ArrayList<>(); surfaces.add(reader.getSurface()); surfaces.add(preview);
            if(depthReader!=null) surfaces.add(depthReader.getSurface());
            device.createCaptureSession(surfaces,new CameraCaptureSession.StateCallback() {
                @Override public void onConfigured(CameraCaptureSession s) {
                    if(stopping) {s.close();return;} session=s;
                    try {
                        CaptureRequest.Builder b=device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
                        for(Surface surface:surfaces) b.addTarget(surface);
                        b.set(CaptureRequest.CONTROL_AF_MODE,CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO);
                        b.set(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF);
                        int[] ois=characteristics.get(CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION);
                        if(contains(ois,CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF)) b.set(CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE,CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF);
                        Range<Integer>[] rates=characteristics.get(CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES);
                        if(rates!=null) for(Range<Integer> rate:rates) if(rate.getLower()==15&&rate.getUpper()==15) { b.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE,rate);break; }
                        s.setRepeatingRequest(b.build(),new CameraCaptureSession.CaptureCallback() {
                            @Override public void onCaptureCompleted(CameraCaptureSession cs,CaptureRequest req,TotalCaptureResult r) { metadata(r); }
                        },handler);
                    } catch(Exception e) { event("camera_error",e.toString()); }
                }
                @Override public void onConfigureFailed(CameraCaptureSession s) {
                    s.close();
                    if(depthReader!=null) { depthReader.close(); depthReader=null; event("depth_unavailable","Color/depth stream combination rejected; retrying color only"); configure(); }
                    else event("camera_error","Capture session configuration failed");
                }
            },handler);
        } catch(Exception e) { event("camera_error",e.toString()); }
    }
    private void metadata(TotalCaptureResult r) {
        try {
            writer.offer(new JSONObject().put("kind","camera_metadata").put("timestamp_ns",r.get(CaptureResult.SENSOR_TIMESTAMP))
                .put("exposure_ns",r.get(CaptureResult.SENSOR_EXPOSURE_TIME)).put("rolling_shutter_skew_ns",r.get(CaptureResult.SENSOR_ROLLING_SHUTTER_SKEW))
                .put("focus_diopters",r.get(CaptureResult.LENS_FOCUS_DISTANCE)).put("af_state",r.get(CaptureResult.CONTROL_AF_STATE))
                .put("intrinsics",array(r.get(CaptureResult.LENS_INTRINSIC_CALIBRATION))));
        } catch(JSONException e) { event("camera_error",e.toString()); }
    }
    private void image(ImageReader source) {
        try(Image im=source.acquireLatestImage()) {
            if(im==null||stopping) return;
            long stamp=im.getTimestamp(); if(stamp-lastFrame<60_000_000L) return; lastFrame=stamp;
            int w=im.getWidth(),h=im.getHeight(); int[] pixels=new int[w*h];
            Image.Plane y=im.getPlanes()[0]; ByteBuffer b=y.getBuffer(); int offset=b.position();
            for(int row=0;row<h;row++) for(int col=0;col<w;col++) {
                int v=b.get(offset+row*y.getRowStride()+col*y.getPixelStride())&255;
                pixels[row*w+col]=0xff000000|(v<<16)|(v<<8)|v;
            }
            Bitmap bitmap=Bitmap.createBitmap(pixels,w,h,Bitmap.Config.ARGB_8888);
            ByteArrayOutputStream bytes=new ByteArrayOutputStream(); bitmap.compress(Bitmap.CompressFormat.JPEG,90,bytes); bitmap.recycle();
            writer.offer(new JSONObject().put("kind","frame").put("timestamp_ns",stamp).put("width",w).put("height",h)
                .put("format","jpeg").put("image_b64",android.util.Base64.encodeToString(bytes.toByteArray(),android.util.Base64.NO_WRAP)));
        } catch(Exception e) { event("camera_error",e.toString()); }
    }
    private void depth(ImageReader source) {
        try(Image im=source.acquireLatestImage()) {
            if(im==null||stopping) return;
            Image.Plane p=im.getPlanes()[0]; ByteBuffer buffer=p.getBuffer(); byte[] data=new byte[buffer.remaining()]; buffer.get(data);
            writer.offer(new JSONObject().put("kind","depth").put("timestamp_ns",im.getTimestamp()).put("width",im.getWidth()).put("height",im.getHeight())
                .put("row_stride",p.getRowStride()).put("pixel_stride",p.getPixelStride()).put("format","DEPTH16")
                .put("data_b64",android.util.Base64.encodeToString(data,android.util.Base64.NO_WRAP)));
        } catch(Exception e) { event("depth_error",e.toString()); }
    }
    public void stop() {
        stopping=true;
        if(session!=null) session.close(); if(device!=null) device.close();
        if(reader!=null) reader.close(); if(depthReader!=null) depthReader.close();
    }
}
