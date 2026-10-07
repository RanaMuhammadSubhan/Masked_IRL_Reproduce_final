"""Inference-only replay of a packaged checkpoint. Writes only to a new output folder."""
import sys
sys.dont_write_bytecode=True
import argparse,pathlib,json,os,types
P=pathlib.Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--run',type=pathlib.Path,required=True);ap.add_argument('--checkpoint',choices=['selected_pretrain.pt','adapted_epoch100.pt'],default='selected_pretrain.pt');ap.add_argument('--endpoint',choices=['pre_adapt_heldout','post_adapt_heldout','strict_zero_shot','pretrain_seen','post_adapt_seen'],default='strict_zero_shot');ap.add_argument('--output',type=pathlib.Path,required=True);ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
expected='adapted_epoch100.pt' if a.endpoint.startswith('post_') else 'selected_pretrain.pt'
assert a.checkpoint==expected,'Choose the checkpoint corresponding to the endpoint'
out=a.output.resolve()
for name in ['results','code','report','evidence','data_manifest','verification','figures','environment']:assert out!=P/name and P/name not in out.parents
assert not out.exists();out.mkdir(parents=True)
os.environ['WANDB_MODE']='disabled';os.environ['GIT_PYTHON_REFRESH']='quiet'
sys.path.insert(0,str(P/'code/reproduction'));sys.path.insert(0,str(P/'code/upstream'))
import torch,numpy as np
from train_v3 import Experiment,FrozenLanguage
from src.models.reward_learning.masked_rl_lang_states_input import FiLMRewardModel
from src.utils.feature_utils import theta_to_language
torch.set_num_threads(4)
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
cfg=read(a.run/'config.json');pr=read(a.run/'protocol.json');d=P/'code/upstream/reconstruction/datasets/simulation_v2'
e=Experiment.__new__(Experiment);e.a=types.SimpleNamespace(method=cfg['method']);e.out=out;e.device=torch.device(a.device)
e.thetas=pr['train_preferences']+pr['adapted_preferences']+pr['strict_preferences'];e.languages=theta_to_language(e.thetas)
e.states_gpu=torch.as_tensor(np.load(d/'states.npy'),device=e.device);e.features=np.load(d/'features.npy');e.masks=torch.as_tensor(np.array([r['mask'] for r in pr['masks']]),device=e.device)
e.pairs={k:np.array(v) for k,v in pr['pairs'].items()};e.lang=FrozenLanguage(e.device)
e.projection=torch.nn.Linear(768,128).to(e.device);e.net=FiLMRewardModel(19,128,[128,256,128]).double().to(e.device)
ck=torch.load(a.run/a.checkpoint,map_location=e.device,weights_only=True);assert ck['t5_revision']==cfg['t5_revision'];e.net.load_state_dict(ck['cost_nn']);e.projection.load_state_dict(ck['projection'])
prefs=range(70,100) if a.endpoint=='strict_zero_shot' else range(40) if a.endpoint.endswith('_seen') else range(40,70)
score=e.evaluate('test',prefs,a.endpoint)
original=np.load(a.run/(a.endpoint+'.npz'));replay=np.load(out/(a.endpoint+'.npz'))
result=dict(run_id=a.run.name,endpoint=a.endpoint,win_rate=score,original_win_rate=float(original['win_rates'].mean()),max_abs_prediction_difference=float(np.max(np.abs(original['predicted_cost']-replay['predicted_cost']))),training_performed=False)
(out/'evaluation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
