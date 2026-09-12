"""Independent RGB-CfC pretraining and causal S1 feature extraction.

Shared encoder never consumes joints, labels, outcomes or future observations.
YOLO cues are disabled for this legacy-value adaptation, explicitly, not faked.
"""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def pretrain(args):
    import numpy as np
    from train_factorized_event_observer_v2 import train, TrainConfig
    with np.load(args.umi, allow_pickle=False) as source:
        if set(source['split'].tolist()) != {'train', 'validation'}:
            raise ValueError('origin-only cache required; external test prohibited')
    return train(args.umi, args.output, TrainConfig(
        steps=args.steps, batch_size=16, window=24, hidden=96,
        learning_rate=1e-4, eval_every=200, seed=20260906,
        use_cues=False, forecast_horizons_s=(.3, .6), rare_event_fraction=.25),
        device=args.device)


def extract(args):
    import cv2
    import numpy as np
    import pandas as pd
    import torch
    from event_rl.factorized_observer import load_observer, CausalWindows, model_inputs
    from event_rl.factorized_events import HEADS
    from event_rl.semantic_cues import sha256
    from torch.utils.data import DataLoader
    from finetune_v4.score_s1 import validate_task_table
    if args.output.exists():
        raise ValueError('refuse overwrite')
    model, saved = load_observer(args.checkpoint, args.device)
    if saved['model_spec']['use_cues'] or saved['config']['window'] != 24:
        raise ValueError('must use new RGB-only 24-step pretraining')
    frames = pd.concat([pd.read_parquet(p) for p in sorted((args.data/'data').rglob('*.parquet'))]).sort_values('index').reset_index(drop=True)
    episodes = pd.concat([pd.read_parquet(p) for p in sorted((args.data/'meta/episodes').rglob('*.parquet'))]).sort_values('episode_index')
    times = pd.read_parquet(args.data/'capture_timestamps.parquet')
    validate_task_table(pd.read_parquet(args.data/'meta/tasks.parquet'))
    for k in ('index', 'episode_index', 'frame_index'):
        if not np.array_equal(frames[k], times[k]):
            raise ValueError('capture time alignment mismatch')
    if len(frames) != 13750 or len(episodes) != 50 or not np.array_equal(frames['index'], np.arange(len(frames))):
        raise ValueError('wrong new S1 dataset')
    all_features = []
    all_states = np.stack(frames['observation.state']).astype(np.float32)
    if all_states.shape != (13750, 25) or not np.isfinite(all_states).all():
        raise ValueError('not finite real S1 states')
    camera = 'observation.images.base_0_rgb'
    source_hashes = {str(p.relative_to(args.data)): sha256(p) for p in sorted((args.data/'data').rglob('*.parquet'))}
    for p in [args.data/'capture_timestamps.parquet', args.data/'conversion_mapping.json', args.data/'meta/info.json']:
        source_hashes[str(p.relative_to(args.data))] = sha256(p)
    for _, ep in episodes.iterrows():
        eid, length = int(ep.episode_index), int(ep.length)
        begin, end = int(ep.dataset_from_index), int(ep.dataset_to_index)
        rows = frames.iloc[begin:end]
        if length != end-begin or not np.all(rows.episode_index == eid) or not np.array_equal(rows.frame_index, np.arange(length)):
            raise ValueError('episode boundaries mismatch')
        prefix = 'videos/' + camera
        path = args.data/prefix/f"chunk-{int(ep[prefix+'/chunk_index']):03d}"/f"file-{int(ep[prefix+'/file_index']):03d}.mp4"
        rel = str(path.relative_to(args.data))
        if rel not in source_hashes:
            source_hashes[rel] = sha256(path)
        capture = cv2.VideoCapture(str(path))
        offset = float(ep[prefix+'/from_timestamp']) * 30
        if abs(offset-round(offset)) > 1e-3 or abs(capture.get(cv2.CAP_PROP_FPS)-30) > .001:
            raise ValueError('invalid video timestamps')
        capture.set(cv2.CAP_PROP_POS_FRAMES, round(offset))
        images, anchor = [], []
        for fi in range(length):
            ok, bgr = capture.read()
            if not ok:
                raise ValueError('video decode failed')
            if fi % 3 == 0:
                rgb = cv2.cvtColor(cv2.resize(bgr, (128,128), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                images.append(np.moveaxis(rgb, -1, 0)[None])
                anchor.append(begin+fi)
        capture.release()
        n = len(anchor)
        arrays = dict(images_uint8=np.stack(images), semantic_features=np.zeros((n,34),np.float32),
                      labels=np.full((n,4),-1,np.int64), attempt_uid=np.full(n,str(eid)),
                      query_id=np.arange(n), elapsed_s=times.iloc[anchor].elapsed_s.to_numpy())
        features = []
        with torch.inference_mode():
            for batch in DataLoader(CausalWindows(arrays,24), batch_size=16):
                result = model(**model_inputs(batch,args.device),return_features=True)
                features.append(torch.cat([result['history_features'], *[result[k].softmax(-1) for k in HEADS]],-1).cpu().numpy())
        # Sample-and-hold: the image used at any action anchor is strictly past/current.
        all_features.append(np.concatenate(features)[np.arange(length)//3])
        print(json.dumps(dict(stage='features',episode=eid,frames=length)),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output, features=np.concatenate(all_features).astype(np.float32),
        states=all_states, index=frames['index'].to_numpy(), episode=frames.episode_index.to_numpy(),
        frame=frames.frame_index.to_numpy(), time=times.elapsed_s.to_numpy(),
        contract_json=json.dumps(dict(format='umi_s1_cfc_features_v1',checkpoint_sha256=sha256(args.checkpoint),
            source_hashes=source_hashes,video_stride=3,window=24,feature_names='96 CfC hidden + 15 predicted event probabilities',
            future_inputs=False,state_in_shared_encoder=False,raw_action_inputs=False)))
    return dict(stage='features_complete',file=str(args.output),sha256=sha256(args.output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['pretrain','extract'])
    parser.add_argument('--umi',type=Path)
    parser.add_argument('--data',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=2000)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    import torch
    torch.set_num_threads(4)
    print(json.dumps((pretrain if args.stage=='pretrain' else extract)(args),ensure_ascii=False),flush=True)
