"""Fresh actor finetune, preserving base normalization except audited static axes."""
from pathlib import Path
import copy
import sys
import json
import argparse

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def preserve_checkpoint_stats(kwargs):
    result=copy.deepcopy(kwargs)
    result.pop('dataset_stats',None)
    for group,step in [('preprocessor_overrides','normalizer_processor'),
                       ('postprocessor_overrides','unnormalizer_processor')]:
        if group in result and step in result[group]:result[group][step].pop('stats',None)
    return result


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--normalization-repair',type=Path,required=True)
    options,forwarded=parser.parse_known_args()
    sys.argv=[sys.argv[0],*forwarded]
    import torch
    from safetensors.torch import load_file
    from lerobot.scripts import lerobot_train as training
    import finetune_v4.train_event_weighted as weighted_module
    from finetune_v4.train_event_weighted import main as train_main
    repair_path=options.normalization_repair
    repair=json.loads(repair_path.read_text())
    if repair['format']!='s1_near_static_normalization_repair_v1':raise ValueError('missing audited normalization repair')
    original_mean=weighted_module.event_weighted_mean

    def checked_loss(losses,weights):
        loss=original_mean(losses,weights)
        if not torch.isfinite(loss) or loss.detach().item()>100:
            raise ValueError('normalized flow loss >100 or nonfinite; abort before backward, inspect input scales')
        return loss

    weighted_module.event_weighted_mean=checked_loss
    original=training.make_pre_post_processors

    def fixed_processors(*,policy_cfg,pretrained_path=None,pretrained_revision=None,**kwargs):
        if not pretrained_path or Path(pretrained_path).parent.name!='050000':
            raise ValueError('must initialize from specified pure050000, not smoke or stopped actor')
        pre,post=original(policy_cfg=policy_cfg,pretrained_path=pretrained_path,
                          pretrained_revision=pretrained_revision,**preserve_checkpoint_stats(kwargs))
        for pipeline,filename in [(pre,'policy_preprocessor_step_5_normalizer_processor.safetensors'),
                                  (post,'policy_postprocessor_step_0_unnormalizer_processor.safetensors')]:
            expected=load_file(str(Path(pretrained_path)/filename))
            states=[s.state_dict() for s in pipeline.steps if hasattr(s,'_tensor_stats')]
            if len(states)!=1 or states[0].keys()!=expected.keys():raise ValueError('processor statistics schema changed')
            if any(not torch.equal(states[0][k].cpu(),v) for k,v in expected.items()):
                raise ValueError('checkpoint normalization was overridden')
            updated={k:v.clone() for k,v in expected.items()}
            for feature,rows in repair['repair'].items():
                for row in rows:
                    i=row['index']
                    for stat in ('mean','std'):
                        if float(expected[feature+'.'+stat][i])!=row['old_'+stat]:
                            raise ValueError('normalization repair belongs to another base')
                        updated[feature+'.'+stat][i]=row[stat]
            step=next(s for s in pipeline.steps if hasattr(s,'_tensor_stats'))
            if step._stats_explicitly_provided:raise ValueError('unexpected explicit processor stats')
            step.load_state_dict(updated)
            actual=step.state_dict()
            if any(not torch.equal(actual[k].cpu(),v) for k,v in updated.items()):
                raise ValueError('normalization repair did not load exactly')
        print('AUDITED_NORMALIZERS verified: base stats except action[1,23,24], state[14,23,24]; raw values/units unchanged',flush=True)
        return pre,post

    training.make_pre_post_processors=fixed_processors
    train_main()


if __name__=='__main__':main()
