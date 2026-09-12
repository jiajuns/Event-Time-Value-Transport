#!/usr/bin/env python3
"""原始 RGB-CfC + 独立 state25 value adapter 的五折 TD(lambda)-AWR。"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch

N=24754; E=100; TASK="pick up the pot lid and place it beside the pot"

def main():
    p=argparse.ArgumentParser()
    for k in ("features","labels","output"): p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--steps",type=int,default=1500); p.add_argument("--device",default="cuda"); a=p.parse_args()
    if a.output.exists(): raise ValueError("refusing overwrite")
    from cfc_value_umi_s1_20260906.value import fit,lambda_returns,sha256,state_histories,transitions
    with np.load(a.features,allow_pickle=False) as z:
        features,states,episode,time,index,frame=(z[k] for k in ("features","states","episode","time","index","frame")); contract=json.loads(str(z["contract_json"].item()))
    if features.shape!=(N,111) or states.shape!=(N,25) or not np.array_equal(index,np.arange(N)) or sorted(np.unique(episode).tolist())!=list(range(E)): raise ValueError("bad pot features")
    if contract["format"]!="pot_lid_original_cfc_features_v1" or contract["task_text"]!=TASK: raise ValueError("wrong feature contract")
    labels=json.loads(a.labels.read_text()); outcomes={int(x["episode_index"]):x["terminal_outcome"] for x in labels["labels"]}
    if labels["format"]!="pot_lid_robot_terminal_labels_v1" or sorted(outcomes)!=list(range(E)): raise ValueError("wrong labels")
    tr=transitions(episode,time,outcomes,chunk=50,rate=.1); histories=state_histories(states,episode,length=8,stride=3)
    a.output.mkdir(parents=True); oof=np.full(N,np.nan,np.float32); adv=oof.copy(); delta=oof.copy(); weights=np.ones(N,np.float32); records=[]
    folds=[x.tolist() for x in np.array_split(np.arange(E),5)]
    for fold,heldout in enumerate(folds):
        train=[e for e in range(E) if e not in heldout]
        model,values,scale,history=fit(features,histories,episode,tr,train,a.steps,20260910+fold,a.device)
        mask=np.isin(episode,heldout); valid=mask&tr["known"]; advantage=lambda_returns(values,tr)-values
        oof[mask]=values[mask]; adv[valid]=advantage[valid]
        td=tr["reward"]+np.where(tr["terminal"],0,tr["gamma"]*values[tr["next"]])-values; delta[valid]=td[valid]
        weights[valid]=np.clip(np.exp(np.clip(advantage[valid]/(3*scale),-5,5)),.9,1.1)
        trainmask=np.isin(episode,train)&tr["known"]; baseline=float(tr["mc"][trainmask].mean())
        record=dict(fold=fold,train_episodes=train,scoring_episodes=heldout,advantage_scale_train_only=scale,
                    heldout_mc_mse=float(np.mean((values[valid]-tr["mc"][valid])**2)),
                    train_constant_baseline_mse=float(np.mean((baseline-tr["mc"][valid])**2)),training_history=history)
        torch.save(dict(format="pot_lid_original_cfc_state25_value_v1",state_dict={k:v.cpu() for k,v in model.state_dict().items()},
                        feature_contract=contract,feature_dim=111,state_history_length=8,state_stride=3,
                        labels_sha256=sha256(a.labels),features_sha256=sha256(a.features),record=record),a.output/f"value_fold{fold}.pt")
        records.append(record); print(json.dumps(dict(stage="fit_cfc",**record)),flush=True)
    if not np.isfinite(oof).all() or not np.isfinite(weights).all() or weights.std()<1e-7: raise ValueError("invalid/uniform crossfit weights")
    last=np.array([np.flatnonzero(episode==e)[-1] for e in range(E)]); weights[last]=1;adv[last]=np.nan;delta[last]=np.nan
    model,_,_,history=fit(features,histories,episode,tr,list(range(E)),a.steps,20261910,a.device)
    torch.save(dict(format="pot_lid_original_cfc_state25_value_v1",state_dict={k:v.cpu() for k,v in model.state_dict().items()},feature_contract=contract,
                    feature_dim=111,state_history_length=8,state_stride=3,scoring_use="not used for current actor weights",training_history=history),a.output/"value_all.pt")
    side=a.output/"event_weights.parquet"
    pd.DataFrame(dict(index=index,episode_index=episode,frame_index=frame,event_weight=weights,value_oof=oof,td_delta_oof=delta,
                      advantage_td_lambda=adv,mc_return=tr["mc"],reward_known=tr["known"],endpoint_index=tr["next"],terminal_chunk=tr["terminal"])).to_parquet(side,index=False)
    receipt=dict(format="pot_lid_original_cfc_crossfit_awr_v1",task_text=TASK,features_sha256=sha256(a.features),labels_sha256=sha256(a.labels),
                 sidecar_sha256=sha256(side),observer_sha256=contract["checkpoint_sha256"],frames=N,episodes=E,chunk=50,folds=records,
                 outcomes={x:sum(v==x for v in outcomes.values()) for x in ("success","failure","unknown")},
                 weight_min=float(weights.min()),weight_max=float(weights.max()),weight_mean=float(weights.mean()),weight_std=float(weights.std()),
                 shared_rgb_cfc_frozen=True,raw_states_only_in_separate_s1_adapter=True,own_episode_critic_leakage=False,online_rl=False,
                 limitations=["The 100 robot demonstrations have success terminal labels only.","This branch learns expert-trajectory progress, not unseen failure classification."])
    (a.output/"receipt.json").write_text(json.dumps(receipt,indent=2)); print(json.dumps(receipt),flush=True)
if __name__=="__main__": torch.set_num_threads(4); main()
