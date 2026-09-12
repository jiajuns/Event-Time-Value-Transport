"""Read-only integrity checks before each actor run; no visual auto-selection."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cfc_value_umi_s1_20260906.value import sha256
from finetune_v4.train_event_weighted import load_event_weights
from finetune_v4.score_s1 import validate_task_table


def run(a):
    receipt=json.loads((a.scores/'receipt.json').read_text())
    if receipt['format']!='umi_s1_crossfit_td_lambda_awr_v1':raise ValueError('not learned crossfit value')
    base=sha256(a.source/'model.safetensors')
    if base!='90f7bc58a75005cbeeb49d06590fb8b3ec4d74ed66d17b9bea4f6613195a938a':raise ValueError('not pure050000 source')
    if a.source.parent.name!='050000':raise ValueError('wrong base step')
    if sha256(a.observer)!=receipt['observer_sha256']:raise ValueError('wrong pretrained video CfC')
    for path,digest in receipt['dataset_source_hashes'].items():
        if sha256(a.data/path)!=digest:raise ValueError('dataset changed: '+path)
    if sha256(a.labels)!=receipt['labels_sha256']:raise ValueError('labels changed')
    if sha256(a.scores/'event_weights.parquet')!=receipt['sidecar_sha256']:raise ValueError('weights changed')
    if sha256(a.repair)!='9f860daca85fa6ed139f9b81dce7e4badef590fe279d951417985a94eb759cc9':raise ValueError('wrong normalization repair')
    validate_task_table(pd.read_parquet(a.data/'meta/tasks.parquet'))
    weights=load_event_weights(a.scores/'event_weights.parquet')
    frames=pd.concat([pd.read_parquet(p,columns=['index','episode_index','frame_index']) for p in sorted((a.data/'data').rglob('*.parquet'))]).sort_values('index')
    sidecar=pd.read_parquet(a.scores/'event_weights.parquet')
    for key in ('index','episode_index','frame_index'):
        if not np.array_equal(frames[key],sidecar[key]):raise ValueError('sidecar identity mismatch')
    if len(weights)!=13750:raise ValueError('wrong frame count')
    for f in receipt['folds']:
        if set(f['train_episodes'])&set(f['scoring_episodes']):raise ValueError('critic own-episode leakage')
    result=dict(status='passed',base_sha256=base,observer_sha256=receipt['observer_sha256'],
        sidecar_sha256=receipt['sidecar_sha256'],frames=len(weights),weight_min=float(weights.min()),
        weight_max=float(weights.max()),actor_input='original RGB/state/task only; no event token',
        recipe='fresh050000, bounded crossfit TDlambda-AWR, audited near-static normalization repair')
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('source','data','scores','observer','labels','repair','output'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
