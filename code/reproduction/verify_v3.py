"""Independent result verifier: no model inference, no test-set retuning."""
import argparse,csv,json,hashlib,itertools
from pathlib import Path
import numpy as np
from io_utils import read,write,sha

def verify(run,repo):
    run=Path(run);repo=Path(repo);cfg=read(run/'config.json');pr=read(run/'protocol.json')
    assert read(run/'status.json')['status']=='COMPLETE'
    assert cfg['pretrain_epochs']==1000 and cfg['adapt_epochs']==100 and cfg['pretrain_lr']==.001 and cfg['adapt_lr']==.0001
    assert cfg['batch_size']==512 and cfg['importance_estimator']=='released_cross_sample' and cfg['t5_frozen']
    assert cfg['masked_loss_weight']==(10 if cfg['method']=='masked' else 0) and cfg['noise_distribution']=='uniform_0_1_paper'
    for rel,h in cfg['source_hashes'].items():assert sha(repo/rel.replace(chr(92), chr(47)))==h
    for name,h in cfg['script_hashes'].items():assert sha(run.parent.parent/'source_archive'/h/name)==h
    d=repo/'reconstruction/datasets/simulation_v2'
    for rel,h in cfg['dataset_hashes'].items():assert sha(d/rel)==h
    scenes=np.load(d/'scene_ids.npy');groups=np.load(d/'start_goal_group_ids.npy');features=np.load(d/'features.npy');unscaled=np.load(d/'features_unscaled.npy')
    split=read(d/'splits.json')['indices'];assert pr['split']==split
    for x,y in itertools.combinations(split,2):
        assert not set(split[x])&set(split[y]);assert not set(scenes[split[x]])&set(scenes[split[y]]);assert not set(groups[split[x]])&set(groups[split[y]])
    scale=read(d/'feature_scaling.json');assert np.allclose(unscaled[split['train']].min(0),scale['min']) and np.allclose(unscaled[split['train']].max(0),scale['max'])
    assert np.allclose(features,(unscaled-scale['min'])/(np.array(scale['max'])-scale['min']))
    raw=np.load(d/'trajectories.npy',mmap_mode='r')
    for k in np.unique(groups):assert np.allclose(raw[groups==k][:,[0,-1],:115],raw[groups==k][0,[0,-1],:115],atol=1e-9,rtol=0)
    train=pr['train_preferences'];adapt=pr['adapted_preferences'];strict=pr['strict_preferences'];alltheta=train+adapt+strict
    assert len(set(map(tuple,train)))==40 and len(set(map(tuple,adapt)))==len(set(map(tuple,strict)))==30
    assert pr['validation_preferences']==train and not set(map(tuple,strict))&set(map(tuple,train+adapt)) and not set(map(tuple,train))&set(map(tuple,adapt))
    rng=np.random.default_rng(cfg['seed']);ti=np.array(split['train'])
    for stage,thetas in [('pretrain',train),('adapt',adapt)]:
        rec=pr['demonstrations'][stage];assert len(rec)==len(thetas)
        for theta,record in zip(thetas,rec):
            z=-20*(features[ti]@theta);p=np.exp(z-z.max());p/=p.sum();expected=rng.choice(ti,10,replace=True,p=p)
            assert record['theta']==theta and record['indices']==expected.tolist()
    # Independently recover corruption draws, without importing the training corruption function.
    saved=read(repo/'config/data_split_config/theta_to_pred_mask_sdim19.json')
    assert sha(repo/'config/data_split_config/theta_to_pred_mask_sdim19.json')=='598ada8219de103109b47c18ebd7a6618fb7fc5b12b5e3f610a4f511e096d36e'
    assert len(saved)==242
    final_manifest=read(run/'strict_zero_shot_preferences.json')
    assert final_manifest['preference_manifest_sha256']=='76461da8b99b505fadceb22dcec44f55d7acf2c097051d03d2e48757ced0b1c6'
    excluded={'_'.join(map(str,t)) for t in train+adapt}
    rank=lambda k:hashlib.sha256(('masked-irl-v3-final-preferences-20261006:'+k).encode()).hexdigest()
    chosen=sorted(set(saved)-excluded,key=rank)[:30]
    assert strict==[list(map(int,k.split('_'))) for k in chosen]
    for key,entry in zip(chosen,final_manifest['entries']):
        assert entry['key']==key and entry['rank_sha256']==rank(key) and entry['mask']==saved[key]
    import sys;sys.path.insert(0,str(repo));from src.utils.feature_utils import theta_to_state_mask
    for theta,record in zip(alltheta,pr['masks']):
        source=np.array(saved['_'.join(map(str,theta))] if cfg['mask_type']=='author_saved_llm_mask' else theta_to_state_mask(theta,state_dim=19)).reshape(-1)
        seed=int.from_bytes(hashlib.sha256((f"v3-corruption:{cfg['seed']}:"+','.join(map(str,theta))).encode()).digest()[:8],'big')
        flips=np.random.default_rng(seed).random(19)<cfg['p_flip'];expected=np.logical_xor(source.astype(bool),flips).astype(float)
        assert np.array_equal(source,record['source_mask']) and np.array_equal(expected,record['mask']) and record['derived_seed']==seed
    frozen=read(run/'t5_verification.json');assert frozen['before_sha256']==frozen['after_sha256'] and frozen['optimizer_excluded'] and frozen['all_parameters_no_grad'] and frozen['backbone_frozen']
    assert frozen['revision']=='a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1' and frozen['cache_equivalence']['status']=='PASS'
    selection=read(run/'selection.json');vals=list(csv.DictReader((run/'validation.csv').open()))
    losses=list(csv.DictReader((run/'losses.csv').open()))
    assert [(r['stage'],int(r['epoch'])) for r in losses]==[('pretrain',i) for i in range(1,1001)]+[('adapt',i) for i in range(1,101)]
    assert np.isfinite([[float(r[k]) for k in ['irl_loss','mask_loss','total_loss']] for r in losses]).all()
    assert [int(r['epoch']) for r in vals]==list(range(10,1001,10))
    chosen=max(vals,key=lambda r:float(r['win_rate']));assert int(chosen['epoch'])==selection['epoch'] and selection['checkpoint_selected_from']=='validation_train_identities'
    lock=read(run/'FINAL_TEST_STARTED.json');assert cfg['config_frozen_timestamp']<selection['selected_timestamp']<lock['final_test_timestamp']
    assert sha(run/'config.json')==lock['config_sha256'] and sha(run/'selected_pretrain.pt')==lock['selected_checkpoint_sha256'] and sha(run/'adapted_epoch100.pt')==lock['adapted_checkpoint_sha256']
    endpoint_results={}
    def score_file(path,prefids,splitname):
        z=np.load(path);assert np.array_equal(z['preference_indices'],prefids);assert np.array_equal(z['pairs'],pr['pairs'][splitname])
        ids=z['indices'];pairs=z['pairs'];assert set(ids)<=set(split[splitname]);assert np.all(groups[pairs[:,0]]==groups[pairs[:,1]])
        expected_pairs=[];splitids=np.array(split[splitname])
        for group in sorted(set(groups[splitids])):expected_pairs.extend(itertools.combinations(splitids[groups[splitids]==group].tolist(),2))
        assert np.array_equal(pairs,expected_pairs) and np.array_equal(ids,np.unique(pairs))
        mapping={v:i for i,v in enumerate(ids)};pair=np.array([[mapping[a],mapping[b]] for a,b in pairs]);values=[]
        for n,pref in enumerate(prefids):
            gt=features[ids]@alltheta[pref];assert np.allclose(gt,z['true_cost'][n]);pred=z['predicted_cost'][n];assert np.isfinite(pred).all()
            delta=gt[pair[:,0]]-gt[pair[:,1]];pd=pred[pair[:,0]]-pred[pair[:,1]];valid=np.abs(delta)>1e-10
            assert int(valid.sum())==int(z['non_tie_counts'][n])
            values.append(float(np.mean(np.where(np.abs(pd[valid])<=1e-10,.5,np.sign(pd[valid])==np.sign(delta[valid])))))
        assert np.allclose(values,z['win_rates'],atol=1e-12);return float(np.mean(values))
    for v in vals:
        s=score_file(run/f"validation_epoch{v['epoch']}.npz",list(range(40)),'validation');assert abs(s-float(v['win_rate']))<1e-12
    metrics=list(csv.DictReader((run/'metrics.csv').open()))
    for label,prefs in [('pre_adapt_heldout',range(40,70)),('post_adapt_heldout',range(40,70)),('strict_zero_shot',range(70,100)),('pretrain_seen',range(40)),('post_adapt_seen',range(40))]:
        s=score_file(run/(label+'.npz'),list(prefs),'test');record=next(float(r['win_rate']) for r in metrics if r['endpoint']==label);assert abs(s-record)<1e-12;endpoint_results[label]=s
    rt=read(run/'runtime.json');assert rt['optimizer_steps']==1100 and rt['examples_seen']==430000
    result=dict(status='PASS',run_id=run.name,method=cfg['method'],seed=cfg['seed'],p_flip=cfg['p_flip'],mask_type=cfg['mask_type'],selected_epoch=selection['epoch'],endpoints=endpoint_results,
      t5_checksums_equal=True,preference_validation_isolated=True,source_and_dataset_hashes_verified=True,all_validation_and_test_metrics_recomputed=True,demonstrations_and_corrupted_masks_independently_checked=True)
    return result
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--repo',type=Path,required=True);a=ap.parse_args();r=verify(a.run,a.repo);write(a.run/'independent_verification.json',r);print(json.dumps(r,indent=2))
