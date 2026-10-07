"""Prespecified post-robustness extensions, isolated from primary V3 artifacts."""
import argparse,concurrent.futures,os,shutil,subprocess,sys,threading
from pathlib import Path
from io_utils import read,write,sha,now
from verify_extension import verify

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--base-root',type=Path,required=True);ap.add_argument('--output-root',type=Path,required=True);ap.add_argument('--stage',choices=['lambda','uncertainty'],required=True);ap.add_argument('--gpus',default='0,1');a=ap.parse_args()
    base=a.base_root.resolve();root=a.output_root.resolve();scripts=Path(__file__).resolve().parent;gpus=a.gpus.split(',')
    for stage in ['core','robustness','llm']:assert read(base/f'{stage}_COMPLETE.json')['status']=='PASS'
    assert sha(a.manifest)=='76461da8b99b505fadceb22dcec44f55d7acf2c097051d03d2e48757ced0b1c6'
    assert read(root/'numerical_tests.json')['status']=='PASS'
    root.mkdir(parents=True,exist_ok=True);(root/'runs').mkdir(exist_ok=True)
    def archive():
        for p in scripts.glob('*.py'):
            q=root/'source_archive'/sha(p)/p.name;q.parent.mkdir(parents=True,exist_ok=True)
            if not q.exists():shutil.copy2(p,q)
    def find(spec):
        variant,lam,p,seed=spec;hits=[]
        for r in (root/'runs').glob('train_*'):
            if not (r/'config.json').exists():continue
            c=read(r/'config.json')
            if (c['variant'],c['masked_loss_weight'],c['p_flip'],c['seed'])!=spec:continue
            if not (r/'status.json').exists():raise RuntimeError('A matching run is already active: '+str(r))
            if read(r/'status.json')['status']=='COMPLETE':hits.append(r)
        assert len(hits)<=1,'Duplicate completed extension candidates';return hits[0] if hits else None
    def paired_check(r):
        c=read(r/'config.json');pr=read(r/'protocol.json');matches=[]
        for b in (base/'runs').glob('train_masked_oracle_p0_seed*'):
            if read(b/'config.json')['seed']==c['seed']:matches.append(b)
        assert len(matches)==1;original=matches[0]
        assert read(r/'initialization.json')==read(original/'initialization.json')
        baseline=read(original/'protocol.json');assert pr['demonstrations']==baseline['demonstrations']
        assert all(pr[k]==baseline[k] for k in ['train_preferences','adapted_preferences','strict_preferences','split'])
        assert c['dataset_hashes']==read(original/'config.json')['dataset_hashes']
        write(r/'paired_baseline_verification.json',dict(status='PASS',baseline_run=original.name,initialization_demonstrations_dataset_preferences_equal=True))
    def job(spec,gpu):
        r=find(spec)
        if r is None:
            archive();variant,lam,p,seed=spec;env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
            cmd=[sys.executable,str(scripts/'train_extension.py'),'--repo',str(a.repo),'--manifest',str(a.manifest),'--output-root',str(root),'--variant',variant,'--mask-lambda',str(lam),'--flip',str(p),'--seed',str(seed)]
            print({'starting':spec,'gpu':gpu,'time':now()},flush=True)
            result=subprocess.run(cmd,env=env)
            if result.returncode:raise RuntimeError('Extension failed; no automatic retry: '+str(spec))
            r=find(spec);assert r is not None
        if not (r/'independent_verification.json').exists():write(r/'independent_verification.json',verify(r,a.repo))
        assert read(r/'independent_verification.json')['status']=='PASS'
        if not (r/'paired_baseline_verification.json').exists():paired_check(r)
        print('VERIFIED '+r.name,flush=True)
    if a.stage=='lambda':specs=[('lambda',lam,0.,seed) for lam in [1.,3.] for seed in [12345,23456,34567]]
    else:
        assert read(root/'lambda_COMPLETE.json')['status']=='PASS'
        specs=[('uncertainty',10.,p,seed) for p in [0.,.05,.1,.2,.3] for seed in [12345,23456,34567]]
    stop=threading.Event()
    def worker(i):
        try:
            for spec in specs[i::len(gpus)]:
                if stop.is_set():raise RuntimeError('Another worker failed; queued jobs stopped')
                job(spec,gpus[i])
        except BaseException:stop.set();raise
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for f in [pool.submit(worker,i) for i in range(len(gpus))]:f.result()
    write(root/f'{a.stage}_COMPLETE.json',dict(status='PASS',time=now(),specs=specs,protocol_sha256=sha(scripts/'EXTENSION_PROTOCOL.md')))
if __name__=='__main__':main()
