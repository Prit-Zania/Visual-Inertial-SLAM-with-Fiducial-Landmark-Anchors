package org.oneplus.slam;

import android.content.Context;
import android.hardware.*;
import android.os.Handler;
import org.json.*;
import java.util.*;

public final class SensorCapture implements SensorEventListener {
    private final SensorManager manager;
    private final Map<Sensor,Integer> keys=new HashMap<>();
    private final SessionWriter writer;
    private final Handler handler;
    private final Map<Integer,Long> counts=new HashMap<>();
    public SensorCapture(Context context,SessionWriter writer,Handler handler) {
        manager=(SensorManager)context.getSystemService(Context.SENSOR_SERVICE);
        this.writer=writer; this.handler=handler;
        int id=0; for(Sensor sensor:manager.getSensorList(Sensor.TYPE_ALL)) keys.put(sensor,++id);
    }
    public static JSONObject inventory(Context context) throws JSONException {
        SensorManager sm=(SensorManager)context.getSystemService(Context.SENSOR_SERVICE);
        JSONArray list=new JSONArray(); JSONObject selected=new JSONObject(); int id=0;
        for(Sensor s:sm.getSensorList(Sensor.TYPE_ALL)) {
            ++id;
            list.put(new JSONObject().put("key",id).put("type",s.getType()).put("name",s.getName())
                .put("vendor",s.getVendor()).put("min_delay_us",s.getMinDelay()).put("resolution",s.getResolution())
                .put("maximum_range",s.getMaximumRange()).put("reporting_mode",s.getReportingMode()));
            if(s.equals(sm.getDefaultSensor(s.getType()))) selected.put(String.valueOf(s.getType()),id);
        }
        ConsumerIrManager ir=(ConsumerIrManager)context.getSystemService(Context.CONSUMER_IR_SERVICE);
        return new JSONObject().put("sensors",list).put("selected_sensors",selected)
            .put("ir_emitter",ir!=null&&ir.hasIrEmitter()).put("ir_range_available",false);
    }
    public void start() {
        for(Sensor s:keys.keySet()) {
            if(s.getReportingMode()!=Sensor.REPORTING_MODE_CONTINUOUS && s.getReportingMode()!=Sensor.REPORTING_MODE_ON_CHANGE) continue;
            int period=(s.getType()==Sensor.TYPE_ACCELEROMETER||s.getType()==Sensor.TYPE_GYROSCOPE)?5000:50000;
            try {
                boolean registered=manager.registerListener(this,s,period,0,handler);
                writer.offer(new JSONObject().put("kind","sensor_registration").put("sensor_key",keys.get(s)).put("registered",registered));
            } catch(Exception e) {
                try { writer.offer(new JSONObject().put("kind","sensor_error").put("sensor_key",keys.get(s)).put("message",e.toString())); }
                catch(JSONException ignored) { }
            }
        }
    }
    public void stop() { manager.unregisterListener(this); }
    @Override public void onSensorChanged(SensorEvent event) {
        try {
            JSONArray values=new JSONArray();
            for(float value:event.values) values.put(Float.isFinite(value)?value:JSONObject.NULL);
            writer.offer(new JSONObject().put("kind","sensor").put("sensor_type",event.sensor.getType())
                .put("sensor_key",keys.get(event.sensor)).put("timestamp_ns",event.timestamp)
                .put("accuracy",event.accuracy).put("values",values));
        } catch(JSONException ignored) { }
    }
    @Override public void onAccuracyChanged(Sensor sensor,int accuracy) { }
}
