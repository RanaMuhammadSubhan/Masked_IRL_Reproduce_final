"""Read-only independent audit of the extracted submission. Never trains models."""
import sys
sys.dont_write_bytecode=True
import argparse,pathlib,hashlib,json,csv,importlib.util,re,zipfile,collections
import numpy as np
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def csvrows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def load(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def dataset(P):
 registry=read(P/'evidence/run_registry.json');c=read(P/registry[0]['path']/'config.json');d=P/'code/upstream/reconstruction/datasets/simulation_v2'
 for name,h in c['dataset_hashes'].items():assert sha(d/name)==h,name
 split=read(d/'splits.json')['indices'];sc=np.load(d/'scene_ids.npy');gr=np.load(d/'start_goal_group_ids.npy');f=np.load(d/'features.npy');u=np.load(d/'features_unscaled.npy');scale=read(d/'feature_scaling.json')
 assert [len(split[k]) for k in ['train','validation','test']]==[1459,359,398]
 assert np.load(d/'states.npy').shape==(2216,21,19)
 import itertools
 for x,y in itertools.combinations(split,2):
  assert not set(split[x])&set(split[y]);assert not set(sc[split[x]])&set(sc[split[y]]);assert not set(gr[split[x]])&set(gr[split[y]])
 assert np.allclose(u[split['train']].min(0),scale['min']) and np.allclose(u[split['train']].max(0),scale['max'])
 assert np.allclose(f,(u-scale['min'])/(np.array(scale['max'])-scale['min']))
 raw=np.load(d/'trajectories.npy',mmap_mode='r')
 for k in np.unique(gr):assert np.allclose(raw[gr==k][:,[0,-1],:115],raw[gr==k][0,[0,-1],:115],atol=1e-9,rtol=0)
 return dict(status='PASS',trajectories=2216,groups=len(np.unique(gr)),dataset_hashes=len(c['dataset_hashes']))
def scan(P):
 machine=re.compile(r'(?:[A-Za-z]:[\\/](?:Users|SSD Data|bld|Research)[\\/]|M:[\\/]|/(?:home|Users)/[A-Za-z][^/\s]*)',re.I)
 secret=re.compile(r'(?:sk-(?:proj-)?[A-Za-z0-9_-]{24,}|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
 findings=[];count=0
 for p in P.rglob('*'):
  if not p.is_file() or '__pycache__' in p.parts:continue
  texts=[]
  if p.suffix.lower() in {'.json','.jsonl','.txt','.md','.py','.csv','.yaml','.yml','.log','.svg','.ps1','.cmd','.xml'}:texts=[p.read_text(encoding='utf-8-sig',errors='replace')]
  elif p.suffix=='.docx':
   with zipfile.ZipFile(p) as z:texts=[z.read(n).decode('utf-8',errors='replace') for n in z.namelist() if n.endswith(('.xml','.rels'))]
  for s in texts:
   count+=1
   for label,pat in [('machine_path',machine),('secret',secret)]:
    if pat.search(s):findings.append(dict(path=p.relative_to(P).as_posix(),kind=label))
 assert not findings,findings
 return dict(status='PASS',text_streams_scanned=count,policy='Pattern scan of text and DOCX XML; no secret-shaped keys or personal absolute paths; not a universal credential detector')
def aggregate(P,recomputed):
 seeds=[12345,23456,34567];primary={};extensions={};checked=0
 for meta in read(P/'evidence/run_registry.json'):
  v=recomputed[meta['run_id']]
  if meta['group']=='primary':primary[(meta['method'],meta['mask_type'],float(meta['p_flip']),meta['seed'])]=v['endpoints']
  else:extensions[(meta['variant'],float(meta['lambda_mask']),float(meta['p_flip']),meta['seed'])]=v['endpoints']
 def check(row,x):
  nonlocal checked
  x=np.array(x);assert len(x)==int(row['n'])==3
  for k,v in [('mean',x.mean()),('sd',x.std(ddof=1)),('se',x.std(ddof=1)/np.sqrt(3))]:assert abs(float(row[k])-v)<1e-9,(row,k,v)
  checked+=1
 for group in ['primary','extensions']:
  folder=P/'results/tables'/group
  for r in csvrows(folder/'per_seed_metrics.csv'):
   assert abs(float(r['win_rate_percent'])-100*recomputed[r['run_id']]['endpoints'][r['endpoint']])<1e-9
  for r in csvrows(folder/'experiment_registry.csv'):
   v=recomputed[r['run_id']];assert int(r['selected_epoch'])==v['selected_epoch']
   for ep in v['endpoints']:
    if ep in r:assert abs(float(r[ep])-100*v['endpoints'][ep])<1e-9
   for key in ['config_path','checkpoint_path','adapted_checkpoint_path','metrics_path','stdout_path','stderr_path']:
    if key in r and r[key]:assert (P/r[key].replace('\\','/')).is_file(),(key,r[key])
  for r in csvrows(folder/'summary_statistics.csv'):
   if group=='primary':v=[100*primary[(r['method'],r['mask_type'],float(r['p_flip']),s)][r['endpoint']] for s in seeds]
   else:v=[100*extensions[(r['variant'],float(r['lambda_mask']),float(r['p_flip']),s)][r['endpoint']] for s in seeds]
   check(r,v)
  for r in csvrows(folder/'paired_method_differences.csv'):
   q=float(r['p_flip']);ep=r['endpoint']
   if group=='primary':
    first,second=r['comparison'].split(' minus ');v=[100*(primary[(first,r['mask_type'],q,s)][ep]-primary[(second,r['mask_type'],q,s)][ep]) for s in seeds]
   else:
    comp=r['comparison'].split(' minus ')[1];v=[100*(extensions[(r['variant'],float(r['lambda_mask']),q,s)][ep]-primary[(comp,'oracle',0 if comp=='lcrl' else q,s)][ep]) for s in seeds]
   check(r,v)
  for r in csvrows(folder/'paired_adaptation_and_corruption_effects.csv'):
   q=float(r['p_flip']);ep=r['endpoint'];v=[]
   for s in seeds:
    if group=='primary':
     x=primary[(r['method'],r['mask_type'],q,s)];clean=primary[(r['method'],r['mask_type'],0,s)]
    else:
     x=extensions[(r['variant'],float(r['lambda_mask']),q,s)];clean=extensions.get((r['variant'],float(r['lambda_mask']),0,s))
    v.append(100*(x['post_adapt_heldout']-x['pre_adapt_heldout'] if ep=='adapted_heldout' else x[ep]-clean[ep]))
   check(r,v)
 return dict(status='PASS',aggregate_and_paired_rows=checked,per_seed_tables_checked=True,registry_paths_checked=True)
def audit(P,hashes=True,progress=True):
 result={'dataset':dataset(P)}
 if hashes:
  manifest=read(P/'DELIVERY_MANIFEST.json')
  for r in manifest['files']:
   p=P/r['path'];assert p.is_file() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],r['path']
  for line in (P/'SHA256SUMS.txt').read_text().splitlines():
   digest,rel=line.split('  ',1);assert sha(P/rel)==digest,rel
  result['hashes']={'status':'PASS','files':len(manifest['files'])}
 result['scan']=scan(P);sys.path.insert(0,str(P/'code/reproduction'))
 core=load('audit_core',P/'code/reproduction/verify_v3.py');ext=load('audit_ext',P/'code/extensions/verify_extension.py')
 registry=read(P/'evidence/run_registry.json');assert len(registry)==60 and len({r['run_id'] for r in registry})==60
 actual={p.name for root in ['primary','extensions'] for p in (P/'results'/root/'runs').iterdir() if p.is_dir()};assert actual=={r['run_id'] for r in registry}
 recomputed={};paired={}
 for i,meta in enumerate(registry,1):
  r=P/meta['path'];cfg=read(r/'config.json');assert cfg['seed'] in [12345,23456,34567]
  v=(core if meta['group']=='primary' else ext).verify(r,P/'code/upstream');recomputed[r.name]=v
  pr=read(r/'protocol.json');init=read(r/'initialization.json')
  base=paired.setdefault(cfg['seed'],(pr,init))
  assert init==base[1] and pr['demonstrations']==base[0]['demonstrations']
  for k in ['train_preferences','adapted_preferences','strict_preferences','split']:assert pr[k]==base[0][k]
  if progress and (i%5==0):print(f'PASS {i}/60 learners',flush=True)
 result['aggregation']=aggregate(P,recomputed);result['status']='PASS';result['completed_learners']=60;result['validation_files_recomputed']=6000;result['final_prediction_files_recomputed']=300;result['runs']=recomputed
 result['scope']='Independent saved-prediction audit; source/dataset/checkpoint hashes; all validation selection scores; masks/confidence/demos; frozen-T5 evidence. Does not rerun checkpoint inference.'
 return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--package',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[1]);ap.add_argument('--report',type=pathlib.Path);ap.add_argument('--dataset-only',action='store_true');ap.add_argument('--skip-manifest',action='store_true',help='Build-time only; final review should not use this flag');a=ap.parse_args();P=a.package.resolve()
 if sys.platform=='win32' and not str(P).startswith('\\\\?\\'):P=pathlib.Path('\\\\?\\'+str(P))
 try:
  r=dataset(P) if a.dataset_only else audit(P,not a.skip_manifest)
  if a.report:a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(r,indent=2),encoding='utf-8')
  print(json.dumps({k:v for k,v in r.items() if k!='runs'},indent=2))
 except Exception as exc:
  print('FAIL: '+repr(exc),file=sys.stderr);raise
if __name__=='__main__':main()
