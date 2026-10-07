"""Portable thin launchers around the completed V3 interfaces. Training is opt-in."""
import sys
sys.dont_write_bytecode=True
import argparse,pathlib,subprocess,json,os,shutil
P=pathlib.Path(__file__).resolve().parents[1]
def run(script,*args):subprocess.run([sys.executable,'-B',str(script),*map(str,args)],check=True)
def protect(path):
 path=path.resolve()
 for name in ['results','code','data_manifest','report','figures','evidence','verification','environment']:
  if path==P/name or P/name in path.parents:raise ValueError('Use a new work directory, outside packaged evidence')
 return path
def main():
 name=pathlib.Path(__file__).stem;ap=argparse.ArgumentParser(description=__doc__)
 if name=='verify_all':
  run(P/'verification/verify_submission.py',*sys.argv[1:]);return
 if name=='verify_dataset':
  run(P/'verification/verify_submission.py','--package',P,'--dataset-only');return
 if name=='check_environment':
  ap.add_argument('--training',action='store_true');ap.add_argument('--fetch-model',action='store_true');a=ap.parse_args()
  import numpy;print('Python',sys.version.split()[0],'NumPy',numpy.__version__)
  if a.training or a.fetch_model:
   import torch,transformers
   print('PyTorch',torch.__version__,'CUDA runtime',torch.version.cuda,'Transformers',transformers.__version__)
   print('CUDA available',torch.cuda.is_available(),'devices',torch.cuda.device_count())
   if a.training:assert torch.cuda.is_available(),'The delivered trainer requires CUDA'
   if a.fetch_model:
    from transformers import T5Tokenizer,T5EncoderModel
    revision='a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1'
    T5Tokenizer.from_pretrained('t5-base',revision=revision);T5EncoderModel.from_pretrained('t5-base',revision=revision)
   from transformers import T5Tokenizer,T5EncoderModel
   T5Tokenizer.from_pretrained('t5-base',revision='a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1',local_files_only=True)
   T5EncoderModel.from_pretrained('t5-base',revision='a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1',local_files_only=True)
  print('PASS');return
 if name=='summarize_results':
  ap.add_argument('--package',type=pathlib.Path,default=P);ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args();out=protect(a.output);out.mkdir(parents=True,exist_ok=False)
  # Recheck raw metrics before exporting byte-identical canonical tables.
  run(P/'verification/verify_submission.py','--package',a.package,'--report',out/'independent_audit.json')
  shutil.copytree(a.package/'results/tables',out/'tables');print(out);return
 ap.add_argument('--output',type=pathlib.Path,required=True);ap.add_argument('--gpus',default='0')
 if name=='run_extensions':ap.add_argument('--base-root',type=pathlib.Path,required=True)
 a=ap.parse_args();out=protect(a.output);repo=P/'code/upstream';manifest=P/'data_manifest/preference_manifest.json';out.mkdir(parents=True,exist_ok=True)
 if name=='run_core_v3':
  run(P/'code/reproduction/test_objective.py')
  for stage in ['one-seed','core']:run(P/'code/reproduction/run_suite.py','--repo',repo,'--manifest',manifest,'--output-root',out,'--stage',stage,'--gpus',a.gpus)
 elif name=='run_corruption':
  for stage in ['robustness','llm']:run(P/'code/reproduction/run_suite.py','--repo',repo,'--manifest',manifest,'--output-root',out,'--stage',stage,'--gpus',a.gpus)
 elif name=='run_extensions':
  run(P/'code/extensions/test_confidence.py');(out/'scripts').mkdir(exist_ok=True)
  protocol=P/'code/extensions/EXTENSION_PROTOCOL.md';dest=out/'scripts/EXTENSION_PROTOCOL.md'
  if dest.exists():assert dest.read_bytes()==protocol.read_bytes()
  else:shutil.copy2(protocol,dest)
  marker=out/'numerical_tests.json'
  if not marker.exists():marker.write_text(json.dumps({'status':'PASS','basis':'test_confidence.py completed successfully before extension launch'}))
  for stage in ['lambda','uncertainty']:run(P/'code/extensions/run_extensions.py','--repo',repo,'--manifest',manifest,'--base-root',a.base_root,'--output-root',out,'--stage',stage,'--gpus',a.gpus)
 else:raise ValueError(name)
if __name__=='__main__':main()
