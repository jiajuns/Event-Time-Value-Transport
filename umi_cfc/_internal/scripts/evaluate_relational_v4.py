#!/usr/bin/env python3
"""Compare relational observers on one shared, partially annotated validation mask."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from event_rl.relational_observer import attach_relation_labels, goal_targets

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def load_events(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        required = ('attempt_uid', 'query_id', 'elapsed_s', 'labels')
        require(all(k in z for k in required), 'events NPZ lacks UID/query/time')
        result = {k: z[k] for k in required}
        result['split'] = z['split'] if 'split' in z else np.full(len(z['query_id']), 'unknown')
    keys = list(zip(result['attempt_uid'].astype(str), result['query_id'].astype(int)))
    require(len(keys) == len(set(keys)), 'events contain duplicate UID/query keys')
    result['keys'] = keys
    return result


def validate_graph(path: Path, events: dict) -> dict:
    with np.load(path, allow_pickle=False) as z:
        row_arrays = [z[k] for k in z.files if z[k].ndim > 0]
        require(row_arrays, 'graph NPZ has no row arrays')
        rows = len(row_arrays[0])
        if 'attempt_uid' in z and 'query_id' in z:
            keys = list(zip(z['attempt_uid'].astype(str), z['query_id'].astype(int)))
            require(keys == events['keys'], 'graph UID/query order differs from events')
        else:
            require(rows == len(events['keys']), 'graph row count differs from events')
        return {'path': str(path), 'sha256': sha256(path), 'rows': rows}


def build_goal_truth(events: dict, annotations: Path) -> dict:
    relation = attach_relation_labels(events, annotations)
    goal = goal_targets(relation, goals=[(0, 1, 1)])[:, 0].astype(np.int8)
    covered = goal >= 0
    return {'goal': goal, 'relation': relation, 'covered': covered,
            'sha256': sha256(annotations), 'path': str(annotations)}


def load_prediction(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        require('attempt_uid' in z and 'query_id' in z, f'{path}: missing keys')
        uid, query = z['attempt_uid'].astype(str), z['query_id'].astype(int)
        keys = list(zip(uid, query))
        require(len(keys) == len(set(keys)), f'{path}: duplicate UID/query keys')
        contract = json.loads(str(z['contract_json'].item())) if 'contract_json' in z else {}
        goals = list(contract.get('goal_schema', {'place_on': None}))
        require('place_on' in goals, f'{path}: no place_on goal')
        goal_column = goals.index('place_on')
        if 'goal_confirmed_ids' in z:
            decision_key = 'goal_confirmed_ids'
        elif 'goal_state_ids' in z:
            decision_key = 'goal_state_ids'
        else:
            raise ValueError(f'{path}: no confirmed/published goal state')
        state = z[decision_key]
        decision = state[:, goal_column] if state.ndim == 2 else state
        # v4 may additionally expose inferred/current-belief states; they are audited,
        # but never substituted for the confirmed decision metric.
        optional = sorted(k for k in z.files if any(token in k for token in
                          ('belief', 'observed_relation', 'relation_observed',
                           'observability', 'inferred')))
    return {'path': path, 'keys': keys, 'lookup': {k: i for i, k in enumerate(keys)},
            'decision': decision.astype(np.int8), 'decision_key': decision_key,
            'optional_v4_fields': optional, 'contract': contract}


def longest_run_seconds(indices: np.ndarray, uid: np.ndarray, times: np.ndarray) -> float:
    longest = 0.0
    for episode in np.unique(uid[indices]):
        t = np.sort(times[indices[uid[indices] == episode]])
        if not len(t):
            continue
        start = previous = t[0]
        steps = np.diff(np.sort(times[uid == episode]))
        nominal = float(np.median(steps[steps > 0])) if np.any(steps > 0) else 0.1
        for current in t[1:]:
            if current - previous > nominal * 1.5 + 1e-9:
                longest = max(longest, previous - start + nominal)
                start = current
            previous = current
        longest = max(longest, previous - start + nominal)
    return longest


def metrics(decision: np.ndarray, truth: np.ndarray, uid: np.ndarray, times: np.ndarray) -> dict:
    known_negative, known_positive, explicit_unknown = truth == 0, truth == 1, truth == 2
    success = decision == 1
    false_points = success & known_negative
    false_episodes = set(uid[false_points].tolist())
    return {
        'covered_sample_points': int((truth >= 0).sum()),
        'known_negative_support_points': int(known_negative.sum()),
        'known_positive_support_points': int(known_positive.sum()),
        'explicit_unknown_ground_truth_points': int(explicit_unknown.sum()),
        'false_success_points_on_known_negative': int(false_points.sum()),
        'false_success_episodes_on_known_negative': len(false_episodes),
        'false_success_episode_uids': sorted(false_episodes),
        'longest_contiguous_false_success_seconds': longest_run_seconds(
            np.flatnonzero(false_points), uid, times),
        'known_positive_recall': (float((success & known_positive).sum() / known_positive.sum())
                                  if known_positive.any() else None),
        'prediction_unknown_fraction_on_covered_mask': float((decision[truth >= 0] == 2).mean()),
        'predicted_success_on_explicit_unknown_ground_truth': int((success & explicit_unknown).sum()),
    }


def evaluate(events_path: Path, graph_path: Path, annotations: Path,
             predictions: list[tuple[str, Path]], output: Path,
             diagnostic_annotations: Path | None = None) -> dict:
    require(not output.exists(), f'refusing to overwrite evaluation: {output}')
    require(len(predictions) >= 1 and len({name for name, _ in predictions}) == len(predictions),
            'prediction names must be nonempty and unique')
    events = load_events(events_path)
    graph = validate_graph(graph_path, events)
    primary = build_goal_truth(events, annotations)
    runs = {name: load_prediction(path) for name, path in predictions}
    common = set(events['keys'])
    for run in runs.values():
        common &= set(run['keys'])
    require(common, 'prediction key intersection is empty')
    event_index = {key: i for i, key in enumerate(events['keys'])}
    ordered = [key for key in events['keys'] if key in common]
    base_indices = np.asarray([event_index[key] for key in ordered])
    split = events['split'].astype(str)[base_indices]
    require(not np.any(np.isin(np.char.lower(split), ['test', 'inference'])),
            'test/inference rows cannot be used for model selection evaluation')
    truth = primary['goal'][base_indices]
    require(np.any(truth >= 0), 'shared annotation coverage is empty')
    uid = events['attempt_uid'].astype(str)[base_indices]
    times = events['elapsed_s'].astype(float)[base_indices]
    reports = {}
    for name, run in runs.items():
        selected = np.asarray([run['lookup'][key] for key in ordered])
        reports[name] = {'prediction': str(run['path']), 'sha256': sha256(run['path']),
                         'confirmed_decision_field': run['decision_key'],
                         'optional_v4_fields_present': run['optional_v4_fields'],
                         **metrics(run['decision'][selected], truth, uid, times)}
    result = {
        'format': 'umi_relational_v4_shared_validation_evaluation_v1',
        'shared_key_count': len(ordered),
        'coverage': {'definition': ('place_on=(unheld,supported,inside): holding comes from reviewed event labels; '
                                    'target relations come from CSV; any known contradiction is evaluable negative, '
                                    'all matches are positive, and explicit unknown without contradiction is unknown'),
                     'covered_points': int((truth >= 0).sum()),
                     'uncovered_common_points_excluded': int((truth < 0).sum()),
                     'partially_annotated_uncovered_points_are_not_failures': True},
        'ground_truth': {k: v for k, v in primary.items() if k not in ('goal', 'relation', 'covered')},
        'runs': reports,
        'events': {'path': str(events_path), 'sha256': sha256(events_path)},
        'graph': graph,
        'selection_safety': {'test_or_inference_rows_present': False,
                             'folder_categories_used_as_ground_truth': False},
        'limitations': ['Metrics use partially covered AI-reviewed validation relations, not test-folder categories.',
                        'Success predictions on explicit unknown ground truth cannot be scored correct or incorrect.'],
    }
    if diagnostic_annotations is not None:
        diagnostic = build_goal_truth(events, diagnostic_annotations)
        diagnostic_truth = diagnostic['goal'][base_indices]
        require(np.any(diagnostic_truth >= 0), 'diagnostic annotation coverage is empty')
        result['additional_development_diagnostic'] = {
            'mixed_into_primary_ground_truth': False,
            'ground_truth': {k: v for k, v in diagnostic.items() if k not in ('goal', 'relation', 'covered')},
            'runs': {name: metrics(run['decision'][np.asarray([run['lookup'][k] for k in ordered])],
                                   diagnostic_truth, uid, times) for name, run in runs.items()},
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    return result


def prediction_arg(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition('=')
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError('prediction must be NAME=PATH')
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--events', required=True, type=Path)
    parser.add_argument('--graph', required=True, type=Path)
    parser.add_argument('--annotations', required=True, type=Path)
    parser.add_argument('--prediction', required=True, action='append', type=prediction_arg)
    parser.add_argument('--diagnostic-annotations', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = evaluate(args.events, args.graph, args.annotations, args.prediction, args.output,
                      args.diagnostic_annotations)
    print(json.dumps({'shared_key_count': result['shared_key_count'],
                      'covered_points': result['coverage']['covered_points']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
