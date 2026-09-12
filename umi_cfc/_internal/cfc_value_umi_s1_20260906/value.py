"""Factual, cross-fitted state value and chunk-aligned TD(lambda)-AWR.

Not Q(s,a), not online RL. Only recorded actions are reweighted. Frozen UMI
RGB-CfC features + separate S1 state-history adapter -> scalar value.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def transitions(episode, time, outcomes, chunk=50, rate=.1):
    """Reward at observed rollout end; truncation/unknown are never failures.

    s_t precedes a_t: actions t..t+49 lead to s_(t+50), NOT s_(t+49).
    Reward is discounted within the terminal chunk, so targets do not depend
    on which overlapping 50-frame offset chain a sample belongs to.
    """
    if chunk <= 0 or rate < 0 or len(episode) != len(time):raise ValueError('invalid transition config')
    n=len(episode)
    nxt=np.full(n,-1,np.int64);gamma=np.zeros(n,np.float32)
    reward=np.full(n,np.nan,np.float32);mc=np.full(n,np.nan,np.float32)
    terminal=np.zeros(n,bool);known=np.zeros(n,bool)
    for ep in np.unique(episode):
        ix=np.flatnonzero(episode==ep)
        if not np.array_equal(ix,np.arange(ix[0],ix[-1]+1)) or np.any(np.diff(time[ix])<=0):
            raise ValueError('noncontiguous episode or invalid timestamps')
        end=ix[-1];endpoint=np.minimum(ix+chunk,end)
        terminal[ix]=endpoint==end
        nxt[ix]=endpoint
        gamma[ix]=np.exp(-rate*(time[endpoint]-time[ix]))
        outcome=outcomes[int(ep)]
        if outcome not in ('success','failure','unknown'):raise ValueError('invalid outcome')
        if outcome!='unknown':
            known[ix]=True
            result=1. if outcome=='success' else -1.
            reward[ix]=np.where(terminal[ix],gamma[ix]*result,0.)
            mc[ix]=np.exp(-rate*(time[end]-time[ix]))*result
    return dict(next=nxt,gamma=gamma,reward=reward,mc=mc,terminal=terminal,known=known)


def lambda_returns(values, tr, lam=.95):
    if not 0<=lam<=1:raise ValueError('invalid lambda')
    result=np.full(len(values),np.nan,np.float32)
    for i in range(len(values)-1,-1,-1):
        if not tr['known'][i]:continue
        if tr['terminal'][i]:result[i]=tr['reward'][i]
        else:
            j=tr['next'][i]
            if j<=i:raise ValueError('noncausal transition ordering')
            result[i]=tr['reward'][i]+tr['gamma'][i]*((1-lam)*values[j]+lam*result[j])
    return result


def state_histories(states, episode, length=8, stride=3):
    histories=np.empty((len(states),length,states.shape[-1]),np.float32)
    for ep in np.unique(episode):
        ix=np.flatnonzero(episode==ep)
        histories[ix]=states[np.maximum(ix[:,None]-np.arange(length-1,-1,-1)[None,:]*stride,ix[0])]
    return histories


class Value(nn.Module):
    def __init__(self, feature_dim, feature_mean, feature_std, state_mean, state_std):
        super().__init__()
        for key,value in [('feature_mean',feature_mean),('feature_std',feature_std),('state_mean',state_mean),('state_std',state_std)]:
            self.register_buffer(key,torch.as_tensor(value,dtype=torch.float32))
        self.s1_adapter=nn.Sequential(nn.Linear(8*25,64),nn.LayerNorm(64),nn.Tanh())
        self.head=nn.Sequential(nn.Linear(feature_dim+64,128),nn.GELU(),nn.Linear(128,1),nn.Tanh())

    def forward(self, features, states):
        f=(features-self.feature_mean)/self.feature_std
        s=((states-self.state_mean)/self.state_std).clamp(-10,10).flatten(1)
        return self.head(torch.cat([f,self.s1_adapter(s)],-1)).squeeze(-1)


def predict(model, features, states):
    with torch.no_grad():
        return torch.cat([model(features[i:i+512],states[i:i+512]) for i in range(0,len(features),512)]).cpu().numpy()


def fit(features, states, episode, tr, train_episodes, steps, seed, device):
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    selected=np.flatnonzero(np.isin(episode,train_episodes)&tr['known'])
    if not len(selected):raise ValueError('no reviewed value supervision')
    fmean=features[selected].mean(0);fstd=np.maximum(features[selected].std(0),.05)
    current=states[selected,-1]
    smean=current.mean(0);sstd=np.maximum(current.std(0),.001)
    model=Value(features.shape[1],fmean,fstd,smean,sstd).to(device)
    target=copy.deepcopy(model).eval()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-3)
    ft=torch.as_tensor(features,device=device);st=torch.as_tensor(states,device=device)
    pools=[np.flatnonzero((episode==ep)&tr['known']) for ep in train_episodes]
    pools=[p for p in pools if len(p)]
    history=[]
    target_returns=tr['mc'].copy()
    for step in range(1,steps+1):
        # MC warm start followed by fitted TD(lambda), frozen target per 50 updates.
        if step>200 and (step-1)%50==0:
            target_returns=lambda_returns(predict(target,ft,st),tr)
        chosen=np.array([rng.choice(pools[rng.integers(len(pools))]) for _ in range(128)])
        batch=torch.as_tensor(chosen,device=device)
        y=torch.as_tensor(target_returns[chosen],device=device)
        loss=(model(ft[batch],st[batch])-y).square().mean()
        if not torch.isfinite(loss):raise ValueError('nonfinite critic loss')
        opt.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step()
        if step%50==0:target.load_state_dict(model.state_dict())
        if step==1 or step%500==0 or step==steps:
            history.append(dict(step=step,loss=float(loss.detach())))
    model.eval()
    values=predict(model,ft,st)
    train_adv=lambda_returns(values,tr)[selected]-values[selected]
    return model,values,max(float(train_adv.std()),.05),history


def run(args):
    import pandas as pd
    if args.output.exists():raise ValueError('refuse overwrite critic run')
    with np.load(args.features,allow_pickle=False) as source:
        features=source['features'];states=source['states'];episode=source['episode'];time=source['time']
        index=source['index'];frame=source['frame'];contract=json.loads(str(source['contract_json'].item()))
    if (features.shape!=(13750,111) or states.shape!=(13750,25) or not np.isfinite(features).all()
            or not np.isfinite(states).all() or not np.array_equal(index,np.arange(13750))):
        raise ValueError('invalid frozen S1 feature contract')
    labels=json.loads(args.labels.read_text())
    outcomes={int(row['episode_index']):row['terminal_outcome'] for row in labels['labels']}
    if len(labels['labels'])!=50 or sorted(outcomes)!=list(range(50)):raise ValueError('labels must cover unique50 episodes')
    if labels['provenance']!='AI visual review of recorded multiview sequences; not human gold labels':
        raise ValueError('missing reward provenance')
    if labels['conversion_mapping_sha256']!=contract['source_hashes']['conversion_mapping.json']:
        raise ValueError('reward labels belong to a different episode mapping')
    tr=transitions(episode,time,outcomes)
    histories=state_histories(states,episode)
    args.output.mkdir(parents=True)
    oof=np.full(len(index),np.nan,np.float32);adv=oof.copy();delta=oof.copy()
    weights=np.ones(len(index),np.float32);folds=[]
    # Five contiguous collection blocks: no episode is scored by its fitted critic.
    for fold in range(5):
        heldout=list(range(fold*10,(fold+1)*10))
        train_episodes=[e for e in range(50) if e not in heldout]
        model,v,scale,history=fit(features,histories,episode,tr,train_episodes,args.steps,20260906+fold,args.device)
        mask=np.isin(episode,heldout);valid=mask&tr['known']
        a=lambda_returns(v,tr)-v
        oof[mask]=v[mask];adv[valid]=a[valid]
        d=tr['reward']+np.where(tr['terminal'],0,tr['gamma']*v[tr['next']])-v
        delta[valid]=d[valid]
        weights[valid]=np.clip(np.exp(np.clip(a[valid]/(3.*scale),-5,5)),.9,1.1)
        trainmask=np.isin(episode,train_episodes)&tr['known']
        mse=float(np.mean((v[valid]-tr['mc'][valid])**2)) if valid.any() else None
        baseline=float(tr['mc'][trainmask].mean())
        baseline_mse=float(np.mean((baseline-tr['mc'][valid])**2)) if valid.any() else None
        record=dict(fold=fold,train_episodes=train_episodes,scoring_episodes=heldout,
                    advantage_scale_train_only=scale,heldout_mc_mse=mse,train_constant_baseline_mse=baseline_mse,training_history=history)
        torch.save(dict(format='umi_rgb_cfc_s1_aux_value_v1',state_dict={k:v.cpu() for k,v in model.state_dict().items()},
            feature_contract=contract,feature_dim=111,state_history_length=8,state_stride=3,
            labels_sha256=sha256(args.labels),features_sha256=sha256(args.features),record=record),args.output/f'value_fold{fold}.pt')
        folds.append(record)
        print(json.dumps(dict(stage='critic',**record)),flush=True)
    if not np.isfinite(oof).all() or not np.isfinite(weights).all() or weights.std()<1e-6:raise ValueError('invalid/constant OOF weights')
    # The action after the last recorded state has no observed effect. It must
    # not receive terminal-outcome credit. Earlier short chunks evaluate only
    # the recorded prefix; padded/unrecorded future actions are not evaluated.
    last=np.array([np.flatnonzero(episode==ep)[-1] for ep in np.unique(episode)])
    weights[last]=1.;adv[last]=np.nan;delta[last]=np.nan
    # Save a final all-reviewed-episode value model for future inference, never score actor with it.
    model,_,_,history=fit(features,histories,episode,tr,list(range(50)),args.steps,20261006,args.device)
    torch.save(dict(format='umi_rgb_cfc_s1_aux_value_v1',state_dict={k:v.cpu() for k,v in model.state_dict().items()},
        feature_contract=contract,feature_dim=111,state_history_length=8,state_stride=3,
        scoring_use='not used for current actor weights; these use held-out-fold critics',training_history=history),args.output/'value_all.pt')
    pd.DataFrame(dict(index=index,episode_index=episode,frame_index=frame,event_weight=weights,
        value_oof=oof,td_delta_oof=delta,advantage_td_lambda=adv,mc_return=tr['mc'],
        reward_known=tr['known'],endpoint_index=tr['next'],terminal_chunk=tr['terminal'])).to_parquet(args.output/'event_weights.parquet',index=False)
    receipt=dict(format='umi_s1_crossfit_td_lambda_awr_v1',features_sha256=sha256(args.features),
        labels_sha256=sha256(args.labels),sidecar_sha256=sha256(args.output/'event_weights.parquet'),
        observer_sha256=contract['checkpoint_sha256'],dataset_source_hashes=contract['source_hashes'],
        frames=len(index),episodes=50,chunk=50,discount_rate_per_second=.1,td_lambda=.95,
        terminal_reward={'success':1.,'failure':-1.,'unknown':'masked; actor weight=1'},
        reward_time='recorded terminal observation; discounted within chunk; not claimed first placement time',
        chunk_boundary='s_t precedes a_t; 50 actions lead to s_(t+50); truncate at last observed state; last action weight=1',
        outcomes={s:sum(v==s for v in outcomes.values()) for s in ('success','failure','unknown')},
        weight_min=float(weights.min()),weight_max=float(weights.max()),weight_mean=float(weights.mean()),weight_std=float(weights.std()),
        folds=folds,shared_encoder_frozen=True,raw_states_only_in_s1_adapter=True,
        test_used=False,candidate_actions_in_value=False,actor_initialization='pure050000',actor_additional_steps=10000,
        limitations=['AI-reviewed terminal labels are not human gold.',
                     'Only observed behavior is evaluated; no counterfactual action ranking or online exploration.',
                     'S1 failure recovery cannot be learned without S1 failures; UMI event negatives do not replace them.',
                     'All-success S1 labels train demonstration-progress value, not S1 failure discrimination.',
                     'Sparse terminal reward; recording-end discount may reward shorter recordings, not exact placement speed.',
                     'Cross-fitting avoids own-episode fitting, not all same-session correlations or calibration error.',
                     'Small clipped AWR updates do not guarantee improvement or prevent forgetting.'])
    (args.output/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--features',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1500);p.add_argument('--device',default='cuda')
    args=p.parse_args();torch.set_num_threads(4)
    print(json.dumps(run(args),ensure_ascii=False),flush=True)
