"""Freeze separately reviewed terminal outcomes and their video mapping."""
import json
from pathlib import Path
from value import sha256

root=Path(__file__).resolve().parent
a=root/'audit_a/terminal_outcomes.json'
b=root/'audit_b/terminal_labels.jsonl'
rows=json.loads(a.read_text())['labels']+[json.loads(s) for s in b.read_text().splitlines() if s.strip()]
rows.sort(key=lambda r:r['episode_index'])
if [r['episode_index'] for r in rows]!=list(range(50)):raise ValueError('missing/duplicate episodes')
data=root.parent/'finetune_huanggua/dataset'
mapping=json.loads((data/'conversion_mapping.json').read_text())
result=dict(format='s1_huanggua_terminal_rewards_v1',
    provenance='AI visual review of recorded multiview sequences; not human gold labels',
    label_source_sha256={str(p.relative_to(root)):sha256(p) for p in (a,b)},
    conversion_mapping_sha256=sha256(data/'conversion_mapping.json'),
    criterion='released cucumber remains supported on tray after gripper withdraws; co-visibility is insufficient',
    timestamp_note='evidence_time_s identifies reviewed evidence; not an exact first-placement boundary',
    labels=rows)
with (root/'terminal_labels.json').open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
print(json.dumps({k:sum(r['terminal_outcome']==k for r in rows) for k in ('success','failure','unknown')}))
