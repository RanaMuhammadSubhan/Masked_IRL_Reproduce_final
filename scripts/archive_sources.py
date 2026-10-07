import argparse,pathlib,hashlib,shutil
ap=argparse.ArgumentParser();ap.add_argument('--source',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args()
for p in a.source.glob('*.py'):
 h=hashlib.sha256(p.read_bytes()).hexdigest();q=a.output/'source_archive'/h/p.name;q.parent.mkdir(parents=True,exist_ok=True)
 if q.exists():assert q.read_bytes()==p.read_bytes()
 else:shutil.copy2(p,q)
