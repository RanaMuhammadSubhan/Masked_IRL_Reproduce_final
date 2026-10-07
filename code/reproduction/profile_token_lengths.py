"""Check exact batch-padding length memoization against the current tokenizer."""
import argparse,os,sys,time,json
from pathlib import Path
import numpy as np
import torch
from types import SimpleNamespace
from train_v3 import FrozenLanguage
from io_utils import read,write
ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
os.environ['HF_HOME']=str(a.repo.parent/'.cache/huggingface');sys.path.insert(0,str(a.repo))
from src.utils.feature_utils import theta_to_language
from transformers import T5Tokenizer
manifest=read(a.manifest);texts=theta_to_language(manifest['train_preferences']+manifest['old_v2_test_preferences_excluded']+manifest['final_test_preferences'])
tok=T5Tokenizer.from_pretrained('t5-base',revision='a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1',local_files_only=True)
unique=list(dict.fromkeys(texts));encoded=tok(unique,padding=False,truncation=True)['input_ids'];lengths={t:len(ids) for t,ids in zip(unique,encoded)}
rng=np.random.default_rng(71007);batches=[list(texts[:40])*10,list(texts[40:70])*10,texts]
batches += [[texts[i] for i in rng.integers(0,100,size=n)] for n in [1,2,19,30,40,300,359,398,400,512] for _ in range(10)]
batches += [[t]*359 for t in texts]
fake=FrozenLanguage.__new__(FrozenLanguage);fake.tokenizer=tok;fake.token_lengths={};fake.cache={};fake.device=torch.device('cpu');fake.equivalence=None
fake.encoder=lambda input_ids,attention_mask:SimpleNamespace(last_hidden_state=torch.stack([input_ids.float(),attention_mask.float()],dim=-1))
for batch in batches:
    original=tok(batch,padding=True,truncation=True,return_tensors='pt');length=max(lengths[t] for t in batch)
    assert original['input_ids'].shape[1]==length
    check=tok(batch,padding='max_length',max_length=length,truncation=True,return_tensors='pt')
    for key in original:assert np.array_equal(original[key].numpy(),check[key].numpy())
    expected=fake.encoder(**original).last_hidden_state.mean(dim=1)
    torch.testing.assert_close(fake.pooled(batch),expected,rtol=0,atol=0)
batch=batches[0];start=time.perf_counter()
for _ in range(100):tok(batch,padding=True,truncation=True,return_tensors='pt')['input_ids'].shape[1]
old=time.perf_counter()-start;start=time.perf_counter()
for _ in range(100):max(lengths[t] for t in batch)
new=time.perf_counter()-start
result=dict(status='PASS',instruction_count=len(texts),distinct_instruction_count=len(unique),tested_batches=len(batches),input_ids_and_attention_masks_bitwise_equal=True,actual_pooled_method_with_token_sensitive_mock_encoder_bitwise_equal=True,old_100_batch_seconds=old,memoized_100_batch_seconds=new,scope='CPU padding-length lookup only; unchanged encoder inputs, pooling, missing-key order, projection and objective')
write(a.output,result);print(json.dumps(result,indent=2))
