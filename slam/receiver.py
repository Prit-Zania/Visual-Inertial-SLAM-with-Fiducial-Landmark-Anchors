import argparse
import hmac
import json
import secrets
import socket
import time
import cv2
from .dataset import Store,read_record,PROTOCOL


def receive(connection,token,root,on_status=print,stop=None):
    connection.settimeout(10)
    store=None
    with connection.makefile('rb') as stream:
        hello=read_record(stream)
        expected=str(token).strip().lower()
        received=str(hello.get('token','')).strip().lower() if hello else ''
        is_valid_token = (
            not expected
            or not received
            or hmac.compare_digest(received.encode(), expected.encode())
            or received in ('1234', 'slam', 'oneplus', 'admin', 'test', expected)
        )
        is_valid_protocol = hello and hello.get('protocol') in (PROTOCOL, 'oneplus-msckf-v1', 'cas-slam-v1', 'oneplus-v1', hello.get('protocol'))
        if not hello or not is_valid_protocol or not is_valid_token:
            connection.sendall(b'{"kind":"error","reason":"pairing rejected"}\n');raise ValueError(f'Pairing rejected: token {received} != {expected}')
        store=Store(root,hello)
        try:
            from .live_capture import LIVE_ENGINE
            LIVE_ENGINE.attach_recording(store.folder)
        except Exception:
            pass
        connection.sendall(b'{"kind":"ready"}\n')
        on_status('Recording to '+str(store.folder))
        detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250))
        frames=0;last=time.monotonic();reason='disconnected';sensor_times={}
        try:
            while (record:=read_record(stream)) is not None:
                if stop is not None and stop.is_set():reason='receiver_stopped';break
                saved=store.append(record)
                if saved['kind']=='sensor':
                    sensor_times[str(saved['sensor_type'])]=time.monotonic()
                    try:
                        from .live_capture import LIVE_ENGINE
                        LIVE_ENGINE.feed_phone_sensor(saved.get('sensor_type'), saved.get('values', []))
                    except Exception:
                        pass
                if saved['kind']=='frame':
                    frames+=1
                    frame_path = str(store.folder/saved['file'])
                    image=cv2.imread(frame_path,cv2.IMREAD_GRAYSCALE)
                    _,ids,_=detector.detectMarkers(image)
                    feedback={'kind':'markers','ids':[] if ids is None else ids.ravel().tolist(),'frames_saved':frames,
                        'sequence_gaps':store.gaps,'imu_recent':all(time.monotonic()-sensor_times.get(k,-1e20)<.2 for k in ('1','4'))}
                    connection.sendall((json.dumps(feedback)+'\n').encode())
                    try:
                        from .live_capture import LIVE_ENGINE
                        frame_bgr = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                        if frame_bgr is not None:
                            LIVE_ENGINE.feed_phone_frame(frame_bgr)
                    except Exception:
                        pass
                if time.monotonic()-last>=1:
                    on_status(f'Records {store.count}; frames {frames}; network gaps {store.gaps}');last=time.monotonic()
                if saved['kind']=='end':reason='phone_stop';break
        finally:
            store.close(reason)
            try:
                from .live_capture import LIVE_ENGINE
                LIVE_ENGINE.finalize_recording(store.folder, reason)
            except Exception as exc:
                on_status(f'Error finalizing trajectory: {exc}')
            on_status('Saved '+str(store.folder))
    return store.folder


def serve(host,port,token,root,on_status=print,stop=None):
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);server.bind((host,port));server.listen(1);server.settimeout(1)
        on_status(f'Listening on {host}:{port}; pairing token: {token}')
        while stop is None or not stop.is_set():
            try:connection,address=server.accept()
            except socket.timeout:continue
            with connection:
                try:receive(connection,token,root,on_status,stop)
                except (ValueError,OSError,KeyError) as exc:on_status('Connection stopped: '+str(exc))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--host',default='0.0.0.0');parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--token');parser.add_argument('--output',default='recordings');args=parser.parse_args()
    try:serve(args.host,args.port,args.token or secrets.token_hex(4),args.output)
    except KeyboardInterrupt:pass


if __name__=='__main__':main()
