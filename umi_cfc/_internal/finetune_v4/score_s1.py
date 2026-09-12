"""Frozen v4 video observer -> bounded, experimental chunk-weighted BC sidecar.

Not a learned V/Q critic. Each belief uses past/current video only; the offline
chunk label may use its recorded endpoint. No actions, joints or event labels
enter the observer. This adapter explicitly changes the camera domain to S1.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from event_rl.evidence_observer import load_observer, publish_evidence, RELATIONS
from event_rl.relational_graph import CausalGraphTracker, NODE_FEATURES, EDGE_FEATURES
from event_rl.relational_visual import CausalPairVisualState, crop_pair_images, PAIR_FEATURE_NAMES
from event_rl.semantic_cues import YOLOECueDetector, sha256
from build_relational_graph_cache_v3 import _crop

V4_SHA = 'bfbc05ddad3e5c771e2b6dabb70edcbbb6edd6ddaaae8ebe55e14933d2b07294'
TASK = 'put the vegetable to the plate'
CAMERA = 'observation.images.base_0_rgb'
INPUTS = ('images', 'node_features', 'node_images_uint8', 'node_mask',
          'edge_features', 'bindings', 'pair_images_uint8', 'pair_features')


def validate_task_table(tasks):
    """Require the sole task while accepting either pandas parquet layout."""
    if len(tasks) != 1 or 'task_index' not in tasks.columns:
        raise ValueError(f'instruction mismatch: {tasks.to_dict(orient="records")}')
    task = tasks['task'].iloc[0] if 'task' in tasks.columns else tasks.index[0]
    if int(tasks['task_index'].iloc[0]) != 0 or task != TASK:
        raise ValueError(f'instruction mismatch: task={task!r}, records={tasks.to_dict(orient="records")}')


def make_windows(rows, times, window=12):
    """Exactly the right-padding/dt convention of EvidenceWindows, no future."""
    batch = {}
    for key in INPUTS:
        value = np.stack([row[key] for row in rows])
        out = np.full((len(rows), window, *value.shape[1:]),
                      -1 if key == 'bindings' else 0, value.dtype)
        for i in range(len(rows)):
            history = value[max(0, i + 1 - window):i + 1]
            out[i, :len(history)] = history
        batch[key] = out
    batch['dt'] = np.zeros((len(rows), window), np.float32)
    batch['mask'] = np.zeros((len(rows), window), bool)
    for i in range(len(rows)):
        history = times[max(0, i + 1 - window):i + 1]
        batch['dt'][i, 1:len(history)] = np.diff(history)
        batch['mask'][i, :len(history)] = True
    return batch


def bounded_weights(relations, goals, observability, nframes, chunk=50):
    """Pre-fixed heuristic potential, NOT calibrated value or RL advantage.

    Unknown reduces reliability, not a failure label. No hard gating/removal.
    All three goal relations AND goal head limit soft placement potential.
    """
    held, support, region = (relations[:, j] for j in range(3))
    placed = np.minimum.reduce((held[:, 0], support[:, 1], region[:, 1], goals[:, 0, 1]))
    carrying = .35 * held[:, 1] + .15 * held[:, 1] * region[:, 1]
    potential = carrying + (1. - carrying) * placed
    reliability = np.minimum(1. - relations[:, :, 2].max(1), 1. - goals[:, 0, 2])
    reliability *= observability.mean(1)
    # Every sampled output is frame 0,3,6,...; floor is strictly causal.
    anchors = np.arange(nframes) // 3
    endpoint = np.minimum(np.arange(nframes) + chunk, nframes - 1)
    endanchors = endpoint // 3
    delta = potential[endanchors] - potential[anchors]
    confidence = np.minimum(reliability[anchors], reliability[endanchors])
    weights = 1. + .1 * confidence * np.tanh(delta / .2)
    if not np.isfinite(weights).all() or weights.min() < .9 - 1e-7 or weights.max() > 1.1 + 1e-7:
        raise ValueError('invalid bounded weights')
    return dict(event_weight=weights, potential=potential[anchors],
                endpoint_potential=potential[endanchors], potential_delta=delta,
                reliability=confidence, observer_frame=anchors * 3,
                endpoint_frame=endpoint, endpoint_observer_frame=endanchors * 3)


def run(args):
    if args.output.exists():
        raise ValueError('refusing to overwrite scoring output')
    if sha256(args.checkpoint) != V4_SHA:
        raise ValueError('not the user-specified frozen latest v4 checkpoint')
    torch.set_num_threads(4)
    model, saved = load_observer(args.checkpoint, args.device)
    contract = saved['input_contract']
    dc = contract['semantic_cues']['detector']
    if (contract['image_size'] != 128 or saved['config']['window'] != 12 or
            contract['relational_graph']['node_features'] != list(NODE_FEATURES) or
            contract['relational_graph']['edge_features'] != list(EDGE_FEATURES) or
            contract['relational_visual']['pair_feature_names'] != list(PAIR_FEATURE_NAMES)):
        raise ValueError('unsupported v4 visual feature contract')
    detector = YOLOECueDetector(args.yolo, dc['prompts'], device=args.device,
                               imgsz=dc['imgsz'], confidence=dc['conf'])
    for key in ('backend', 'version', 'weights_sha256', 'prompts', 'imgsz', 'conf'):
        if detector.provenance[key] != dc[key]:
            raise ValueError(f'YOLO contract mismatch: {key}')
    info = json.loads((args.data / 'meta/info.json').read_text())
    manifest_path = getattr(args, 'dataset_contract', None)
    manifest = json.loads(manifest_path.read_text()) if manifest_path else None
    if manifest and (manifest.get('format') != 's1_event_finetune_dataset_v1' or
                     manifest.get('info_sha256') != sha256(args.data/'meta/info.json') or
                     manifest.get('task_text') != TASK or manifest.get('fps') != 30):
        raise ValueError('invalid explicit new-dataset contract')
    expected_episodes = manifest['total_episodes'] if manifest else 100
    expected_frames = manifest['total_frames'] if manifest else 21437
    train_episodes = manifest['train_episodes'] if manifest else [i for i in range(100) if i%10!=9]
    holdout_episodes = manifest['drift_holdout_episodes'] if manifest else list(range(9,100,10))
    if (sorted(train_episodes + holdout_episodes) != list(range(expected_episodes)) or
            set(train_episodes) & set(holdout_episodes)):
        raise ValueError('invalid episode partition')
    if info['fps'] != 30 or info['total_episodes'] != expected_episodes or info['total_frames'] != expected_frames:
        raise ValueError('unexpected S1 dataset')
    tasks = pd.read_parquet(args.data / 'meta/tasks.parquet')
    validate_task_table(tasks)
    data_files = sorted((args.data / 'data').rglob('*.parquet'))
    episode_files = sorted((args.data / 'meta/episodes').rglob('*.parquet'))
    frames = pd.concat([pd.read_parquet(p, columns=['index', 'episode_index', 'frame_index', 'timestamp']) for p in data_files])
    frames = frames.sort_values('index').reset_index(drop=True)
    episodes = pd.concat([pd.read_parquet(p) for p in episode_files]).sort_values('episode_index')
    if not np.array_equal(frames['index'], np.arange(expected_frames)):
        raise ValueError('dataset indices are not unique/exhaustive')
    capture_path = args.data/'capture_timestamps.parquet'
    capture_times = None
    if manifest:
        if not capture_path.is_file() or sha256(capture_path) != manifest.get('capture_timestamps_sha256'):
            raise ValueError('missing or wrong recorded capture timestamps')
        capture_times = pd.read_parquet(capture_path)
        for key in ('index','episode_index','frame_index'):
            if not np.array_equal(capture_times[key],frames[key]):
                raise ValueError('recorded capture time identity mismatch')
    args.output.mkdir(parents=True)
    hashes = {str(p.relative_to(args.data)): sha256(p) for p in
              [*data_files, *episode_files, args.data/'meta/info.json', args.data/'meta/tasks.parquet']}
    if manifest:
        hashes[str(capture_path.relative_to(args.data))] = sha256(capture_path)
        hashes[str(manifest_path.relative_to(args.data))] = sha256(manifest_path)
    scored, summaries = [], []
    started = time.monotonic()
    for _, ep in episodes.iterrows():
        eid, length = int(ep['episode_index']), int(ep['length'])
        if args.max_episodes and eid >= args.max_episodes:
            break
        ef = frames[frames.episode_index == eid].copy()
        if len(ef) != length or not np.array_equal(ef.frame_index, np.arange(length)):
            raise ValueError('episode/frame alignment mismatch')
        begin, stop = int(ep['dataset_from_index']), int(ep['dataset_to_index'])
        if (stop - begin != length or
                not np.array_equal(ef['index'].to_numpy(), np.arange(begin, stop))):
            raise ValueError('episode/global-index alignment mismatch')
        if not np.allclose(ef.timestamp, np.arange(length) / 30., atol=2e-5, rtol=0):
            raise ValueError('timestamp alignment mismatch')
        prefix = f'videos/{CAMERA}'
        video = args.data / prefix / f"chunk-{int(ep[prefix+'/chunk_index']):03d}" / f"file-{int(ep[prefix+'/file_index']):03d}.mp4"
        relative = str(video.relative_to(args.data))
        if relative not in hashes:
            hashes[relative] = sha256(video)
        offset = float(ep[prefix + '/from_timestamp']) * 30
        if abs(offset - round(offset)) > 1e-3:
            raise ValueError('video offset not an exact frame')
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened() or abs(capture.get(cv2.CAP_PROP_FPS) - 30) > .001:
            raise ValueError('cannot decode expected 30fps video')
        capture.set(cv2.CAP_PROP_POS_FRAMES, round(offset))
        if abs(capture.get(cv2.CAP_PROP_POS_FRAMES) - round(offset)) > .1:
            raise ValueError('video seek failed')
        tracker, visual = CausalGraphTracker(max_nodes=8), CausalPairVisualState(max_history_s=.3)
        rows, times, detections = [], [], []
        for fi in range(length):
            ok, bgr = capture.read()
            if not ok:
                raise ValueError(f'decode failed episode={eid} frame={fi}')
            if fi % 3:
                continue
            with torch.inference_mode():
                result = detector.model.predict(source=bgr, **detector.options)[0]
            boxes = result.boxes
            det = [] if boxes is None else [dict(role=detector.roles[int(c)], confidence=float(s), xyxy=b.tolist())
                   for b, s, c in zip(boxes.xyxyn.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy())]
            # Degenerate clipped boxes are invalid graph inputs, record rather than invent geometry.
            det = [d for d in det if d['xyxy'][2] > d['xyxy'][0] and d['xyxy'][3] > d['xyxy'][1]]
            features, mask, box, ids, edges, bindings = tracker.update(det)
            node_images = np.zeros((8, 3, 64, 64), np.uint8)
            for slot in np.flatnonzero(mask):
                node_images[slot] = _crop(bgr, box[slot], 64)
            t = float(capture_times.iloc[begin+fi].elapsed_s) if capture_times is not None else float(ef.timestamp.iloc[fi])
            pair_features, crops = visual.update(t, features, mask, box, bindings)
            rgb = cv2.cvtColor(cv2.resize(bgr, (128,128), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            rows.append(dict(images=np.moveaxis(rgb, -1, 0)[None], node_features=features,
                        node_images_uint8=node_images, node_mask=mask, edge_features=edges,
                        bindings=bindings, pair_images_uint8=crop_pair_images(bgr, crops), pair_features=pair_features))
            times.append(t)
            detections.append(dict(episode_index=eid, frame_index=fi, index=int(ef['index'].iloc[fi]),
                                   timestamp=t, detections=det))
        capture.release()
        batches = make_windows(rows, np.asarray(times))
        predictions = {'relations': [], 'current': [], 'goals': [], 'observability': []}
        with torch.inference_mode():
            for start in range(0, len(rows), 16):
                batch = {k: torch.from_numpy(v[start:start+16]).to(args.device) for k,v in batches.items()}
                output = model(**batch)
                predictions['relations'].append(torch.stack([output['relations'][k].softmax(-1) for k in RELATIONS],1).cpu().numpy())
                predictions['current'].append(torch.stack([output['current_relations'][k].softmax(-1) for k in RELATIONS],1).cpu().numpy())
                predictions['goals'].append(output['goals'].softmax(-1).cpu().numpy())
                predictions['observability'].append(output['observability'].sigmoid().cpu().numpy())
        pred = {k: np.concatenate(v) for k,v in predictions.items()}
        stable, inferred, confirmed = publish_evidence(np.asarray(times),
                {k:pred['relations'][:,j] for j,k in enumerate(RELATIONS)},
                {k:pred['current'][:,j] for j,k in enumerate(RELATIONS)}, pred['observability'], pred['goals'])
        values = bounded_weights(pred['relations'], pred['goals'], pred['observability'], length)
        for key, value in values.items():
            ef[key] = value
        ef['endpoint_index'] = begin + ef['endpoint_frame'].to_numpy()
        if not np.array_equal(ef['endpoint_index'].to_numpy(),
                              np.minimum(ef['index'].to_numpy() + 50, stop - 1)):
            raise ValueError('chunk endpoint/global-index alignment mismatch')
        # Fixed episode holdout only for post-finetune drift, NOT unseen by base actor.
        ef['split'] = 'drift_holdout' if eid in holdout_episodes else 'train'
        scored.append(ef)
        with (args.output/f'episode_{eid:03d}.npz').open('xb') as stream:
            np.savez_compressed(stream, **pred, elapsed_s=times, relation_stable=stable,
                                goal_inferred=inferred, goal_confirmed=confirmed)
        with (args.output/f'detections_{eid:03d}.json').open('x') as stream:
            json.dump(detections, stream)
        summary = dict(episode=eid, frames=length, observations=len(rows),
                       weight_min=float(ef.event_weight.min()), weight_max=float(ef.event_weight.max()),
                       confirmed_positive=int((confirmed[:,0]==1).sum()),
                       target_detection_observations=sum(any(d['role']=='target' for d in row['detections']) for row in detections))
        summaries.append(summary)
        print(json.dumps(dict(**summary, elapsed_seconds=round(time.monotonic()-started,1))), flush=True)
    table = pd.concat(scored).sort_values('index')
    if not args.max_episodes and (len(table) != expected_frames or table.event_weight.std() < 1e-7):
        raise ValueError('incomplete or effectively uniform event weights')
    path = args.output/'event_weights.parquet'
    table.to_parquet(path, index=False)
    receipt = dict(format='s1_frozen_v4_bounded_event_bc_v1', complete=not bool(args.max_episodes),
        checkpoint_sha256=sha256(args.checkpoint), sidecar_sha256=sha256(path), source_hashes=hashes,
        scoring_code_sha256=sha256(Path(__file__)), detector=detector.provenance,
        camera=CAMERA, observer_training_camera='umi', cross_embodiment_validated=False,
        observer_quality_gate_passed=False, calibrated_value=False, learned_advantage=False,
        task_text=TASK, event_text_in_task=False, joints_or_actions_used_by_observer=False,
        fps=30, score_hz=10, window=12, chunk_size=50,
        alignment='per-episode floor(frame/3); endpoint=min(frame+50,length-1); no cross-episode',
        potential='c=.35*p(held)+.15*p(held)*p(inside); z=min(p(unheld),p(supported),p(inside),p(goal)); P=c+(1-c)*z',
        weight='1+.1*min(reliability[t],reliability[end])*tanh((P[end]-P[t])/.2)',
        reliability='min(1-max_relation_unknown,1-goal_unknown)*mean_observability',
        uncertainty_is_failure=False, weight_bounds=[.9,1.1], batch_weight_normalization=False,
        train_episodes=train_episodes, drift_holdout_episodes=holdout_episodes,
        holdout_seen_by_base_actor=manifest.get('holdout_seen_by_base_actor',False) if manifest else True,
        observer_dt_source='recorded_capture_timestamps' if manifest else 'nominal_dataset_timestamps',
        dataset_contract_sha256=sha256(manifest_path) if manifest_path else None,
        umi_test_used=False, policy_deployment_authorized=False,
        frame_count=len(table), weight_stats=table.event_weight.describe().to_dict(), episodes=summaries)
    with (args.output/'receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(dict(complete=receipt['complete'], frames=len(table), weights=receipt['weight_stats'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('data','checkpoint','yolo','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--max-episodes',type=int,default=0)
    parser.add_argument('--dataset-contract',type=Path)
    run(parser.parse_args())
