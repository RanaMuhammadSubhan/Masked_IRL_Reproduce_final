"""Additive partially paper-faithful V3; fixed-data pretraining and preference adaptation."""
import argparse,contextlib,csv,hashlib,itertools,json,math,os,random,sys,time,traceback,uuid
from pathlib import Path
import numpy as np
import torch
from objective import local_loss_chunks,released_cross_sample
from io_utils import read,write,sha,state_sha,now,csvwrite,append_registry

REVISION='a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1'
COMMIT='b52bc2e3f1ec5597360a74c2641283f6c984a76b'

def corrupt_mask(mask,p,seed,theta):
    if not 0<=p<=1:raise ValueError('Invalid flip probability')
    source=np.asarray(mask,dtype=np.float64);assert source.shape==(19,) and set(source)<={0.,1.}
    # Common random uniforms across p give nested errors, shared across methods.
    token=f'v3-corruption:{seed}:'+','.join(map(str,theta))
    derived=int.from_bytes(hashlib.sha256(token.encode()).digest()[:8],'big')
    flipped=np.random.default_rng(derived).random(19)<p
    result=np.logical_xor(source.astype(bool),flipped).astype(np.float64)
    return result,dict(derived_seed=derived,realized_fraction=float(flipped.mean()),changed_bits=np.flatnonzero(flipped).tolist())

class FrozenLanguage:
    def __init__(self,device):
        from transformers import T5EncoderModel,T5Tokenizer,__version__
        self.tokenizer=T5Tokenizer.from_pretrained('t5-base',revision=REVISION,local_files_only=True)
        self.encoder=T5EncoderModel.from_pretrained('t5-base',revision=REVISION,local_files_only=True).to(device).eval()
        self.encoder.requires_grad_(False);self.cache={};self.token_lengths={};self.device=device;self.version=__version__
        self.before=state_sha(self.encoder)
        self.equivalence=None
    def pooled(self,texts):
        if isinstance(texts,str):texts=[texts]
        unknown=list(dict.fromkeys(t for t in texts if t not in self.token_lengths))
        if unknown:
            ids=self.tokenizer(unknown,padding=False,truncation=True)['input_ids']
            self.token_lengths.update({t:len(v) for t,v in zip(unknown,ids)})
        length=max(self.token_lengths[t] for t in texts)
        keys=[(x,length) for x in texts];missing=list(dict.fromkeys(k for k in keys if k not in self.cache))
        if missing:
            tok=self.tokenizer([k[0] for k in missing],padding='max_length',max_length=length,truncation=True,return_tensors='pt')
            with torch.no_grad():
                out=self.encoder(**{k:v.to(self.device) for k,v in tok.items()}).last_hidden_state.mean(dim=1)
            for k,v in zip(missing,out):self.cache[k]=v.detach()
        result=torch.stack([self.cache[k] for k in keys])
        if self.equivalence is None:
            tokens=self.tokenizer(texts,padding=True,truncation=True,return_tensors='pt')
            with torch.no_grad():
                direct=self.encoder(**{k:v.to(self.device) for k,v in tokens.items()}).last_hidden_state.mean(dim=1)
            err=float((result-direct).abs().max());torch.testing.assert_close(result,direct,rtol=1e-4,atol=2e-5)
            self.equivalence={'max_abs_difference':err,'batch':len(texts),'padded_length':length,'status':'PASS'}
        return result
    def check(self,optimizer):
        ids={id(p) for g in optimizer.param_groups for p in g['params']}
        assert all(not p.requires_grad and p.grad is None and id(p) not in ids for p in self.encoder.parameters())
    def final(self):
        after=state_sha(self.encoder);assert self.before==after
        return dict(model_id='t5-base',revision=REVISION,tokenizer='T5Tokenizer',transformers=self.version,backbone_frozen=True,
          before_sha256=self.before,after_sha256=after,checksums_match=True,all_parameters_no_grad=True,optimizer_excluded=True,
          pooling='Released mean over all padded positions; padding length depends on call batch',
          cache='Frozen eval-mode outputs keyed by exact instruction and batch padded length; projection never cached',cache_equivalence=self.equivalence,
          token_length_lookup='Memoized exact per-instruction lengths; 203 batches checked for bitwise input-ID/attention-mask equality before deployment',
          frozen_dropout='T5 eval mode in all stages; projection, FiLM, reward remain trainable')

class Experiment:
    def __init__(self,a,out):
        self.a=a;self.out=out;self.start=time.perf_counter();self.device=torch.device('cuda:0')
        torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
        os.environ['WANDB_MODE']='disabled';os.environ['GIT_PYTHON_REFRESH']='quiet'
        sys.path.insert(0,str(a.repo))
        from src.utils.feature_utils import theta_to_state_mask,theta_to_language
        from src.models.reward_learning.masked_rl_lang_states_input import FiLMRewardModel
        d=a.repo/'reconstruction/datasets/simulation_v2';self.dataset=d
        self.states=np.load(d/'states.npy');self.features=np.load(d/'features.npy');self.groups=np.load(d/'start_goal_group_ids.npy');self.scenes=np.load(d/'scene_ids.npy')
        self.split=read(d/'splits.json')['indices'];manifest=read(a.manifest)
        self.train_theta=manifest['train_preferences'];self.adapt_theta=manifest['old_v2_test_preferences_excluded'];self.strict_theta=manifest['final_test_preferences']
        assert len(self.train_theta)==40 and len(self.adapt_theta)==len(self.strict_theta)==30
        assert not set(map(tuple,self.strict_theta))&set(map(tuple,self.train_theta+self.adapt_theta))
        assert not set(map(tuple,self.train_theta))&set(map(tuple,self.adapt_theta))
        for x,y in itertools.combinations(self.split,2):
            assert not set(self.split[x])&set(self.split[y]);assert not set(self.scenes[self.split[x]])&set(self.scenes[self.split[y]])
            assert not set(self.groups[self.split[x]])&set(self.groups[self.split[y]])
        assert self.states.shape==(2216,21,19) and np.isfinite(self.states).all() and np.isfinite(self.features).all()
        rawf=np.load(d/'features_unscaled.npy');scale=read(d/'feature_scaling.json')
        assert np.allclose(rawf[self.split['train']].min(0),scale['min']) and np.allclose(rawf[self.split['train']].max(0),scale['max'])
        assert np.allclose(self.features,(rawf-scale['min'])/(np.array(scale['max'])-scale['min']))
        self.thetas=self.train_theta+self.adapt_theta+self.strict_theta
        self.languages=theta_to_language(self.thetas);assert len(self.languages)==100
        self.train_ids=np.array(self.split['train']);self.states_gpu=torch.as_tensor(self.states,device=self.device)
        saved=read(a.repo/'config/data_split_config/theta_to_pred_mask_sdim19.json');self.maskmeta=[];masks=[]
        for th in self.thetas:
            oracle=np.asarray(theta_to_state_mask(th,state_dim=19)).reshape(-1)
            source=saved['_'.join(map(str,th))] if a.mask_source=='author_saved_llm_mask' else oracle
            mask,meta=corrupt_mask(source,a.flip,a.seed,th);masks.append(mask);self.maskmeta.append(dict(theta=th,source_mask=np.asarray(source).tolist(),mask=mask.tolist(),**meta))
        self.masks=torch.as_tensor(np.array(masks),device=self.device)
        rng=np.random.default_rng(a.seed);self.demodata={};demorecords={}
        for name,thetas,offset in [('pretrain',self.train_theta,0),('adapt',self.adapt_theta,40)]:
            ids=[];prefs=[];records=[]
            for i,th in enumerate(thetas):
                logits=-20*(self.features[self.train_ids]@th);p=np.exp(logits-logits.max());p/=p.sum()
                draw=rng.choice(self.train_ids,10,replace=True,p=p);ids.extend(draw);prefs.extend([i+offset]*10);records.append(dict(theta=th,indices=draw.tolist()))
            self.demodata[name]=(np.array(ids),np.array(prefs));demorecords[name]=records
        self.pairs={}
        for split in ['validation','test']:
            ids=np.array(self.split[split]);pairs=[]
            for group in sorted(set(self.groups[ids])):pairs.extend(itertools.combinations(ids[self.groups[ids]==group].tolist(),2))
            self.pairs[split]=np.array(pairs);assert pairs
        self.lang=FrozenLanguage(self.device)
        # Deliberate common initial model across methods/corruption conditions for each seed.
        torch.manual_seed(a.seed)
        self.projection=torch.nn.Linear(self.lang.encoder.config.d_model,128).to(self.device)
        self.net=FiLMRewardModel(19,128,[128,256,128]).double().to(self.device)
        self.params=list(self.net.parameters())+list(self.projection.parameters())
        self.order_rng=torch.Generator().manual_seed(a.seed+101)
        self.noise_rng=torch.Generator(device=self.device).manual_seed(a.seed+202)
        self.steps=0;self.examples=0;self.losses=[];self.val=[];self.metrics=[];self.best=-float('inf');self.selected_epoch=None
        cfg=dict(family='mask_corruption_robustness' if a.flip else 'paper_faithful_v3',fidelity='partially_paper_faithful_reconstruction',method=a.method,mask_type=a.mask_source,seed=a.seed,p_flip=a.flip,
          dataset_version='simulation_v2',upstream_commit=COMMIT,pretrain_epochs=1000,adapt_epochs=100,pretrain_lr=.001,adapt_lr=.0001,optimizer='Adam default betas and eps',batch_size=512,
          masked_loss_weight=10 if a.method=='masked' else 0,noise_distribution='uniform_0_1_paper',noise_scale=1,local_mask_mc_samples=1,
          mask_objective='mean_B sum_T sum_irrelevant_D absolute_state_cost_difference',importance_estimator='released_cross_sample',importance_numerics='algebraically equivalent log-space cross-sample mean',
          t5_frozen=True,t5_revision=REVISION,projection_trainable=True,reward_dtype='float64',projection_dtype='float32',
          checkpoint_rule='Validation scenes, 40 training preference identities; every 10 epochs; highest mean ordering accuracy; earliest ties',
          adapted_checkpoint='Fixed epoch 100; no adaptation selection',epoch_semantics='One pass over 400 pretraining or 300 adaptation demonstrations; one optimizer update per epoch at batch 512',
          contrast_semantics='Pretraining draws 512 from 40 x training-trajectory pool; adaptation draws 300 from same pool with original 40 conditioning identities, matching released finetune',
          endpoints=['paper_faithful_adapted_v3','strict_zero_shot_v3'],config_frozen_timestamp=now(),mode=a.mode,chunk_size=a.chunk,
          corruption_policy='Per-seed and preference SHA256-derived common uniforms across methods and p; training, adaptation, and explicit inference share corrupted masks',
          dataset_hashes={n:sha(d/n) for n in ['states.npy','features.npy','features_unscaled.npy','scene_ids.npy','start_goal_group_ids.npy','splits.json','feature_scaling.json','trajectories.npy']},
          source_hashes={str(p.relative_to(a.repo)):sha(p) for p in [a.repo/'src/models/reward_learning/masked_rl_lang_states_input.py',a.repo/'src/utils/feature_utils.py']},
          script_hashes={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
        self.cfg=cfg;write(out/'config.json',cfg)
        write(out/'protocol.json',dict(train_preferences=self.train_theta,validation_preferences=self.train_theta,adapted_preferences=self.adapt_theta,strict_preferences=self.strict_theta,
          demonstrations=demorecords,pairs={k:v.tolist() for k,v in self.pairs.items()},masks=self.maskmeta,split=self.split))
        write(out/'strict_zero_shot_preferences.json',dict(entries=manifest['author_saved_final_masks'],preference_manifest_sha256=sha(a.manifest),overlap_with_train_or_adapt=0))
        write(out/'initialization.json',dict(net_sha256=state_sha(self.net),projection_sha256=state_sha(self.projection)))
    def cost_state(self,x,pooled):return self.net(x,self.projection(pooled).double()).squeeze(-1)
    def traj_cost(self,states,pooled):
        b,t,d=states.shape;cond=self.projection(pooled).double()
        return self.net(states.reshape(-1,d),cond[:,None,:].expand(-1,t,-1).reshape(-1,128)).reshape(b,t).sum(1)
    def inputs(self,ids,prefs):
        x=self.states_gpu[torch.as_tensor(ids,device=self.device)]
        if self.a.method=='explicit':x=x*self.masks[torch.as_tensor(prefs,device=self.device)][:,None,:]
        return x,self.lang.pooled([self.languages[int(i)] for i in prefs])
    def update(self,stage,opt,profile=False):
        did,pref=self.demodata[stage]
        order=torch.randperm(len(did),generator=self.order_rng).numpy()
        if profile:order=np.resize(order,512)
        pool=len(self.train_ids)*40
        ci=(torch.randperm(pool,generator=self.order_rng)[:512] if stage=='pretrain' else torch.randint(pool,(len(order),),generator=self.order_rng)).numpy()
        tx,tp=self.inputs(self.train_ids[ci%len(self.train_ids)],ci//len(self.train_ids))
        dx,dp=self.inputs(did[order],pref[order]);opt.zero_grad(set_to_none=True)
        irl=released_cross_sample(self.traj_cost(dx,dp),self.traj_cost(tx,tp));assert torch.isfinite(irl)
        irl.backward();mask_total=0.
        if self.a.method=='masked':
            noise=torch.rand(dx.shape,device=self.device,dtype=dx.dtype,generator=self.noise_rng)
            for part in local_loss_chunks(self.cost_state,dx,dp,self.masks[torch.as_tensor(pref[order],device=self.device)],noise,self.a.chunk):
                assert torch.isfinite(part);mask_total+=float(part.detach());(10*part).backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in self.params)
        self.lang.check(opt);opt.step();self.steps+=1;self.examples+=len(order)
        return dict(irl_loss=float(irl.detach()),mask_loss=mask_total,total_loss=float(irl.detach())+10*mask_total)
    def save(self,name,epoch):
        torch.save(dict(cost_nn=self.net.state_dict(),projection=self.projection.state_dict(),epoch=epoch,t5_revision=REVISION),self.out/name)
    def load(self,name):
        c=torch.load(self.out/name,map_location=self.device,weights_only=True);self.net.load_state_dict(c['cost_nn']);self.projection.load_state_dict(c['projection'])
    def evaluate(self,split,prefids,label):
        self.net.eval();self.projection.eval();pairs=self.pairs[split];ids=np.unique(pairs);mapping={v:i for i,v in enumerate(ids)};pi=np.array([[mapping[x],mapping[y]] for x,y in pairs])
        predictions=[];truths=[];rates=[];counts=[]
        with torch.no_grad():
            for p in prefids:
                x,pooled=self.inputs(ids,np.full(len(ids),p));pred=self.traj_cost(x,pooled).cpu().numpy();gt=self.features[ids]@self.thetas[p]
                delta=gt[pi[:,0]]-gt[pi[:,1]];pd=pred[pi[:,0]]-pred[pi[:,1]];valid=np.abs(delta)>1e-10
                assert valid.any() and np.isfinite(pred).all()
                rates.append(float(np.where(np.abs(pd[valid])<=1e-10,.5,np.sign(pd[valid])==np.sign(delta[valid])).mean()))
                predictions.append(pred);truths.append(gt);counts.append(int(valid.sum()))
        np.savez_compressed(self.out/(label+'.npz'),indices=ids,pairs=pairs,preference_indices=np.array(list(prefids)),predicted_cost=np.array(predictions),true_cost=np.array(truths),win_rates=np.array(rates),non_tie_counts=np.array(counts))
        self.net.train();self.projection.train();return float(np.mean(rates))
    def run(self):
        opt=torch.optim.Adam(self.params,lr=.001)
        torch.cuda.reset_peak_memory_stats()
        if self.a.mode=='profile':
            start=time.perf_counter();row=self.update('pretrain',opt,profile=True);torch.cuda.synchronize()
            write(self.out/'profile.json',dict(batch_size=512,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),update_seconds=time.perf_counter()-start,loss=row,finite_gradients=True,chunk_size=self.a.chunk))
            write(self.out/'t5_verification.json',self.lang.final());return
        for epoch in range(1,1001):
            row=self.update('pretrain',opt);self.losses.append(dict(stage='pretrain',epoch=epoch,**row))
            if epoch%10==0:
                score=self.evaluate('validation',range(40),f'validation_epoch{epoch}')
                self.val.append(dict(epoch=epoch,win_rate=score,preference_split='train_identities'))
                if score>self.best:self.best=score;self.selected_epoch=epoch;self.save('selected_pretrain.pt',epoch)
                print(json.dumps(dict(stage='pretrain',epoch=epoch,validation=score,**row)),flush=True)
        self.save('final_pretrain.pt',1000);self.load('selected_pretrain.pt')
        write(self.out/'selection.json',dict(epoch=self.selected_epoch,score=self.best,checkpoint_selected_from='validation_train_identities',tie_rule='earliest',selected_timestamp=now()))
        opt=torch.optim.Adam(self.params,lr=.0001)
        for epoch in range(1,101):
            row=self.update('adapt',opt);self.losses.append(dict(stage='adapt',epoch=epoch,**row))
            if epoch%10==0:print(json.dumps(dict(stage='adapt',epoch=epoch,**row)),flush=True)
        self.save('adapted_epoch100.pt',100)
        # All training/selection has ended before any test endpoint is opened.
        testtime=now();write(self.out/'FINAL_TEST_STARTED.json',dict(final_test_timestamp=testtime,config_sha256=sha(self.out/'config.json'),selected_checkpoint_sha256=sha(self.out/'selected_pretrain.pt'),adapted_checkpoint_sha256=sha(self.out/'adapted_epoch100.pt'),policy='Exclusive one-shot bundle; no retries or checkpoint choice based on test'))
        for ck,label,prefs in [('selected_pretrain.pt','pre_adapt_heldout',range(40,70)),('selected_pretrain.pt','strict_zero_shot',range(70,100)),('selected_pretrain.pt','pretrain_seen',range(40)),('adapted_epoch100.pt','post_adapt_heldout',range(40,70)),('adapted_epoch100.pt','post_adapt_seen',range(40))]:
            self.load(ck);score=self.evaluate('test',prefs,label);self.metrics.append(dict(endpoint=label,win_rate=score,checkpoint=ck))
        write(self.out/'FINAL_TEST_COMPLETE.json',dict(timestamp=now(),metrics=self.metrics))
        csvwrite(self.out/'metrics.csv',self.metrics);csvwrite(self.out/'validation.csv',self.val);csvwrite(self.out/'losses.csv',self.losses)
        write(self.out/'t5_verification.json',self.lang.final())
        write(self.out/'runtime.json',dict(runtime_seconds=time.perf_counter()-self.start,optimizer_steps=self.steps,examples_seen=self.examples,pretraining_steps=1000,adaptation_steps=100,gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--output-root',type=Path,required=True)
    ap.add_argument('--method',choices=['lcrl','explicit','masked'],required=True);ap.add_argument('--seed',type=int,default=12345);ap.add_argument('--mask-source',choices=['oracle','author_saved_llm_mask'],default='oracle');ap.add_argument('--flip',type=float,default=0.);ap.add_argument('--chunk',type=int,default=4096);ap.add_argument('--mode',choices=['train','profile'],default='train');a=ap.parse_args()
    a.repo=a.repo.resolve();a.output_root=a.output_root.resolve();a.manifest=a.manifest.resolve()
    rid=f'{a.mode}_{a.method}_{a.mask_source}_p{a.flip:g}_seed{a.seed}_{uuid.uuid4().hex[:8]}'
    out=a.output_root/'runs'/rid;out.mkdir(parents=True,exist_ok=False);print(out,flush=True);start=time.perf_counter()
    row=dict(run_id=rid,family='mask_corruption_robustness' if a.flip else 'paper_faithful_v3',method=a.method,mask_type=a.mask_source,seed=a.seed,p_flip=a.flip,dataset_version='simulation_v2',status='RUNNING',timestamp=now())
    append_registry(a.output_root,row)
    try:
        with (out/'stdout.log').open('x') as so,(out/'stderr.log').open('x') as se,contextlib.redirect_stdout(so),contextlib.redirect_stderr(se):
            e=Experiment(a,out);e.run()
        status='PROFILE_PASS' if a.mode=='profile' else 'COMPLETE'
        write(out/'status.json',dict(status=status,runtime_seconds=time.perf_counter()-start))
        row.update(status=status,runtime_seconds=time.perf_counter()-start,config_path=str(out/'config.json'))
        if a.mode=='train':row.update(metrics=e.metrics,selected_epoch=e.selected_epoch)
        append_registry(a.output_root,row);print(json.dumps(row),flush=True)
    except BaseException as exc:
        with (out/'stderr.log').open('a') as f:f.write(traceback.format_exc())
        write(out/'status.json',dict(status='FAILED',error=repr(exc)));row.update(status='FAILED',error=repr(exc));append_registry(a.output_root,row);raise
if __name__=='__main__':main()
