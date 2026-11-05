"""Shared bounded JSONL protocol for Wi-Fi capture and phone-file recovery."""
import base64
import json
import math
import time
import uuid
from pathlib import Path
import cv2
import numpy as np

PROTOCOL='oneplus-msckf-v1'
MAX_LINE=8_000_000


def read_record(stream):
    line=stream.readline(MAX_LINE+1)
    if not line:return None
    if len(line)>MAX_LINE or not line.endswith(b'\n'):raise ValueError('Oversized or truncated record')
    obj=json.loads(line,parse_constant=lambda s:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    if not isinstance(obj,dict):raise ValueError('Record must be an object')
    return obj


def validate(record):
    kind=record.get('kind')
    if kind in ('sensor','frame','camera_metadata','depth'):
        if type(record.get('timestamp_ns')) is not int or record['timestamp_ns']<0:raise ValueError('Invalid timestamp')
    if kind=='sensor':
        for field in ('sensor_type','sensor_key','accuracy'):
            if type(record.get(field)) is not int:raise ValueError('Invalid sensor field')
        values=record.get('values')
        if not isinstance(values,list) or not 1<=len(values)<=64:raise ValueError('Invalid sensor values')
        if any(v is not None and (type(v) not in (int,float) or not math.isfinite(v)) for v in values):raise ValueError('Invalid sensor number')
    if kind in ('frame','depth'):
        for dimension in ('width','height'):
            if type(record.get(dimension)) is not int or not 1<=record[dimension]<=4096:raise ValueError('Invalid image size')


class Store:
    def __init__(self,root,hello):
        if hello.get('protocol')!=PROTOCOL:raise ValueError('Protocol mismatch; use the new Camera2 app')
        try:
            uuid.UUID(str(hello['session_id']))
        except Exception:
            hello['session_id'] = str(uuid.uuid4())
        self.folder=Path(root)/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
        self.folder.mkdir(parents=True,exist_ok=False);(self.folder/'frames').mkdir();(self.folder/'depth').mkdir()
        self.hello={k:v for k,v in hello.items() if k!='token'}
        (self.folder/'metadata.json').write_text(json.dumps(self.hello,indent=2),encoding='utf-8')
        self.stream=(self.folder/'events.jsonl').open('w',encoding='utf-8')
        self.last_seq=0;self.gaps=0;self.count=0

    def append(self,record):
        validate(record); record=dict(record)
        seq=record.get('seq')
        if type(seq) is not int or seq<=self.last_seq:raise ValueError('Non-increasing sequence')
        self.gaps+=seq-self.last_seq-1;self.last_seq=seq
        if record['kind']=='frame':
            raw_b64 = record.pop('image_b64', None) or record.pop('data', None)
            if not raw_b64: raise ValueError('Missing image payload')
            data=base64.b64decode(raw_b64,validate=True)
            image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape!=(record['height'],record['width']):raise ValueError('Image dimensions do not match payload')
            filename=f'frames/{seq:010d}.jpg';(self.folder/filename).write_bytes(data);record['file']=filename
        elif record['kind']=='depth':
            data=base64.b64decode(record.pop('data_b64'),validate=True)
            if len(data)>MAX_LINE or record.get('row_stride',0)<record['width']*2:raise ValueError('Invalid depth stride')
            filename=f'depth/{seq:010d}.bin';(self.folder/filename).write_bytes(data);record['file']=filename
        record['receive_unix_ns']=time.time_ns()
        self.stream.write(json.dumps(record,allow_nan=False)+'\n');self.count+=1
        if self.count%100==0:self.stream.flush()
        return record

    def close(self,reason='complete'):
        self.stream.close()
        (self.folder/'capture_summary.json').write_text(json.dumps({'records':self.count,'sequence_gaps':self.gaps,'reason':reason},indent=2),encoding='utf-8')


def import_phone(path,root):
    store=None
    try:
        with Path(path).open('rb') as stream:
            store=Store(root,read_record(stream))
            while (record:=read_record(stream)) is not None:store.append(record)
        store.close();return store.folder
    except Exception:
        if store is not None:store.close('import_error_partial_data')
        raise


def events(folder):
    folder=Path(folder)
    with (folder/'events.jsonl').open('rb') as stream:
        records=[]
        while (record:=read_record(stream)) is not None:records.append(record)
    return records
