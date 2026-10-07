import csv,hashlib,json,os,time
from pathlib import Path
from datetime import datetime,timezone
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def state_sha(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def csvwrite(path,rows):
    with Path(path).open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def append_registry(folder,row):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);lock=folder/'registry.lock';start=time.time()
    while True:
        try:lock.mkdir();break
        except FileExistsError:
            if time.time()-start>30:raise TimeoutError('Registry lock timeout')
            time.sleep(.1)
    try:
        p=folder/'registry.jsonl'
        with p.open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    finally:lock.rmdir()
