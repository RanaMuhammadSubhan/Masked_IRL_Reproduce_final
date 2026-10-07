"""Gated V3 scheduler: one heavy process/GPU; core precedes downstream robustness."""
import argparse,concurrent.futures,json,os,subprocess,sys,time,shutil
from pathlib import Path
from io_utils import read,write,append_registry,now,sha
from verify_v3 import verify

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--output-root',type=Path,required=True)
    ap.add_argument('--stage',choices=['one-seed','core','robustness','llm'],required=True);ap.add_argument('--gpus',default='0,1');a=ap.parse_args();gpus=a.gpus.split(',')
    root=a.output_root.resolve();scripts=Path(__file__).resolve().parent;root.mkdir(parents=True,exist_ok=True)
    assert a.manifest.is_file(),f'Missing preference manifest: {a.manifest}'
    assert sha(a.manifest)=='76461da8b99b505fadceb22dcec44f55d7acf2c097051d03d2e48757ced0b1c6'
    def archive_sources():
        for script in scripts.glob('*.py'):
            archive=root/'source_archive'/sha(script)/script.name
            archive.parent.mkdir(parents=True,exist_ok=True)
            if not archive.exists():shutil.copy2(script,archive)
    archive_sources()
    def find(method,seed,p,source):
        hits=[]
        for r in (root/'runs').glob('train_*'):
            if not (r/'status.json').exists() or not (r/'config.json').exists():continue
            c=read(r/'config.json')
            if (c['method'],c['seed'],c['p_flip'],c['mask_type'])==(method,seed,p,source) and read(r/'status.json')['status']=='COMPLETE':hits.append(r)
        if len(hits)>1:raise RuntimeError('Multiple completed candidates require explicit audit: '+str(hits))
        return hits[0] if hits else None
    def check(r):
        if not (r/'independent_verification.json').exists():write(r/'independent_verification.json',verify(r,a.repo))
        assert read(r/'independent_verification.json')['status']=='PASS'
    def job(spec,gpu):
        method,seed,p,source=spec;r=find(*spec)
        if r is None:
            archive_sources()
            before=set((root/'runs').glob('*'));env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
            cmd=[sys.executable,str(scripts/'train_v3.py'),'--repo',str(a.repo),'--manifest',str(a.manifest),'--output-root',str(root),'--method',method,'--seed',str(seed),'--flip',str(p),'--mask-source',source]
            print(json.dumps({'starting':spec,'gpu':gpu,'time':now()}),flush=True)
            result=subprocess.run(cmd,env=env)
            if result.returncode:raise RuntimeError(f'Training failed: {spec}; no automatic retry')
            r=find(*spec)
            if r is None:raise RuntimeError('No completed output after successful process')
        check(r);print(json.dumps({'verified':r.name}),flush=True);return r
    core=[(m,s,0.,'oracle') for s in [12345,23456,34567] for m in ['lcrl','explicit','masked']]
    if a.stage!='one-seed':
        for spec in core[:3]:
            r=find(*spec)
            if r is None:raise RuntimeError('One-seed gate incomplete')
            check(r)
    if a.stage in ['robustness','llm']:
        for spec in core:
            r=find(*spec)
            if r is None:raise RuntimeError('Core multi-seed gate incomplete')
            check(r)
    if a.stage=='one-seed':
        runs=[job(spec,gpus[0]) for spec in core[:3]]
        lines=['# V3 one-seed gate','', 'Status: PASS. Protocol correctness, not method ranking, determines this gate.','',
            '| Method | Pre-adapt held-out | Post-adapt held-out | Strict zero-shot | Selected epoch |','|---|---:|---:|---:|---:|']
        for r in runs:
            v=read(r/'independent_verification.json');e=v['endpoints'];lines.append(f"| {v['method']} | {100*e['pre_adapt_heldout']:.2f}% | {100*e['post_adapt_heldout']:.2f}% | {100*e['strict_zero_shot']:.2f}% | {v['selected_epoch']} |")
        lines+=['','All 1,000+100 schedules complete; frozen T5 hashes equal; local Uniform masking/lambda checked; final predictions independently recomputed. Strict identities never enter demonstrations or selection. Adapted identities do receive gradients. Full checks are in each independent_verification.json.']
        (root/'V3_ONE_SEED_GATE.md').write_text('\n'.join(lines));print('\n'.join(lines),flush=True);return
    if a.stage=='core':specs=core[3:]
    elif a.stage=='robustness':specs=[(m,s,p,'oracle') for p in [.05,.1,.2,.3] for s in [12345,23456,34567] for m in ['explicit','masked']]
    else:
        for p in [.05,.1,.2,.3]:
            for s in [12345,23456,34567]:
                for m in ['explicit','masked']:
                    r=find(m,s,p,'oracle')
                    if r is None:raise RuntimeError('Robustness tier incomplete; LLM tier locked')
                    check(r)
        specs=[(m,s,0.,'author_saved_llm_mask') for s in [12345,23456,34567] for m in ['explicit','masked']]
    # Fixed balanced queues, planned without looking at scores. Masked updates
    # are heavier; placing them first avoids one GPU receiving every masked job.
    queues=[[] for _ in gpus];weights=[0 for _ in gpus]
    for spec in sorted(specs,key=lambda s:s[0]!='masked'):
        i=min(range(len(gpus)),key=lambda j:weights[j]);queues[i].append(spec)
        weights[i]+=2 if spec[0]=='masked' else 1
    plan=dict(stage=a.stage,gpus=gpus,queues=queues,weight_policy='Masked=2, other=1; longest jobs first; score-independent')
    planfile=root/f'{a.stage}_gpu_plan.json'
    if not planfile.exists():write(planfile,plan)
    def worker(index):return [job(spec,gpus[index]) for spec in queues[index]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures=[pool.submit(worker,i) for i in range(len(gpus))]
        for f in futures:f.result()
    marker=root/f'{a.stage}_COMPLETE.json'
    if marker.exists():
        old=read(marker);assert old['status']=='PASS' and old['specs']==[list(s) for s in specs]
    else:write(marker,dict(status='PASS',time=now(),specs=specs))
if __name__=='__main__':main()
