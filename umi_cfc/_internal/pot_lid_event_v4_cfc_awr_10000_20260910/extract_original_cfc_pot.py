#!/usr/bin/env python3
"""从锅盖机器人头部视角提取原始 RGB-CfC 因果特征；state25 单独保存。"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import cv2,numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader

CODE=Path(__file__).resolve().parent; sys.path.insert(0,str(CODE))
TASK="pick up the pot lid and place it beside the pot"; N=24754; E=100
CFC_SHA="0b9638af3dd39a39d2f6fe61c6f671c08176e4316f424c1233d4a59f86bb8b89"

def main():
    p=argparse.ArgumentParser()
    for k in ("data","checkpoint","alignment","output"): p.add_argument("--"+k,type=Path,required=True)
    p.add_argument("--device",default="cuda"); a=p.parse_args()
    if a.output.exists(): raise ValueError("refusing overwrite")
    from event_rl.factorized_observer import load_observer,CausalWindows,model_inputs
    from event_rl.factorized_events import HEADS
    from event_rl.semantic_cues import sha256
    if sha256(a.checkpoint)!=CFC_SHA: raise ValueError("wrong original RGB-CfC")
    model,saved=load_observer(a.checkpoint,a.device)
    if saved["model_spec"]["use_cues"] or saved["config"]["window"]!=24: raise ValueError("wrong CfC contract")
    contract=json.loads((a.alignment/"observer_contract.json").read_text())
    if contract["task_text"]!=TASK or contract["total_frames"]!=N or contract["total_episodes"]!=E: raise ValueError("wrong pot contract")
    frames=pd.concat([pd.read_parquet(x) for x in sorted((a.data/"data").rglob("*.parquet"))]).sort_values("index").reset_index(drop=True)
    episodes=pd.concat([pd.read_parquet(x) for x in sorted((a.data/"meta/episodes").rglob("*.parquet"))]).sort_values("episode_index")
    times=pd.read_parquet(a.alignment/"capture_timestamps.parquet")
    if len(frames)!=N or len(episodes)!=E or not np.array_equal(frames["index"],np.arange(N)): raise ValueError("wrong pot dataset")
    for k in ("index","episode_index","frame_index"):
        if not np.array_equal(frames[k],times[k]): raise ValueError("timestamp alignment mismatch")
    states=np.stack(frames["observation.state"]).astype(np.float32)
    if states.shape!=(N,25) or not np.isfinite(states).all(): raise ValueError("invalid state25")
    camera="observation.images.base_0_rgb"; pieces=[]; hashes={}
    for _,ep in episodes.iterrows():
        eid,length=int(ep.episode_index),int(ep.length); begin,end=int(ep.dataset_from_index),int(ep.dataset_to_index)
        rows=frames.iloc[begin:end]
        if len(rows)!=length or not np.array_equal(rows.frame_index,np.arange(length)): raise ValueError("bad episode")
        prefix="videos/"+camera
        video=a.data/prefix/f"chunk-{int(ep[prefix+'/chunk_index']):03d}"/f"file-{int(ep[prefix+'/file_index']):03d}.mp4"
        hashes[str(video.relative_to(a.data))]=sha256(video)
        cap=cv2.VideoCapture(str(video)); offset=float(ep[prefix+"/from_timestamp"])*30
        if not cap.isOpened() or abs(offset-round(offset))>1e-3 or abs(cap.get(cv2.CAP_PROP_FPS)-30)>.001: raise ValueError("bad video")
        cap.set(cv2.CAP_PROP_POS_FRAMES,round(offset)); images=[]; anchors=[]
        for fi in range(length):
            ok,bgr=cap.read()
            if not ok: raise ValueError(f"decode failed {eid}/{fi}")
            if fi%3==0:
                rgb=cv2.cvtColor(cv2.resize(bgr,(128,128),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
                images.append(np.moveaxis(rgb,-1,0)[None]); anchors.append(begin+fi)
        cap.release(); count=len(anchors)
        arrays=dict(images_uint8=np.stack(images),semantic_features=np.zeros((count,34),np.float32),labels=np.full((count,4),-1,np.int64),
                    attempt_uid=np.full(count,str(eid)),query_id=np.arange(count),elapsed_s=times.iloc[anchors].elapsed_s.to_numpy())
        feature=[]
        with torch.inference_mode():
            for batch in DataLoader(CausalWindows(arrays,24),batch_size=16):
                result=model(**model_inputs(batch,a.device),return_features=True)
                feature.append(torch.cat([result["history_features"],*[result[k].softmax(-1) for k in HEADS]],-1).cpu().numpy())
        sample=np.concatenate(feature); pieces.append(sample[np.arange(length)//3]); print(json.dumps(dict(stage="cfc_features",episode=eid,frames=length)),flush=True)
    feature=np.concatenate(pieces).astype(np.float32)
    if feature.shape!=(N,111) or not np.isfinite(feature).all(): raise ValueError("bad features")
    c=dict(format="pot_lid_original_cfc_features_v1",task_text=TASK,checkpoint_sha256=CFC_SHA,source_hashes=hashes,
           camera=camera,video_stride=3,window=24,feature_dim=111,future_inputs=False,state_in_shared_encoder=False,raw_action_inputs=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,features=feature,states=states,index=frames["index"].to_numpy(),episode=frames.episode_index.to_numpy(),
                        frame=frames.frame_index.to_numpy(),time=times.elapsed_s.to_numpy(),contract_json=json.dumps(c))
    print(json.dumps(dict(stage="complete",sha256=sha256(a.output),shape=list(feature.shape))),flush=True)
if __name__=="__main__": torch.set_num_threads(4); main()
