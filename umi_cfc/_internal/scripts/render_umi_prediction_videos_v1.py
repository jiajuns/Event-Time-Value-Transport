#!/usr/bin/env python3
"""Render observer predictions plus exact cached YOLO boxes, never labels."""
import argparse
import bisect
import html
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from event_rl.factorized_events import HEADS
from event_rl.semantic_cues import sha256

RELATION_ZH = {'held_by_actor': '抓持关系', 'supported_by_target': '目标承托', 'in_target_region': '承载区归属'}
RELATION_LABEL_ZH = {'unheld': '未抓持', 'held': '抓持', 'unsupported': '无目标承托',
                     'supported': '由目标承托', 'outside': '区域外', 'inside': '区域内', 'unknown': '不确定'}
GOAL_ZH = {'place_on': '放到目标上', 'maintain_hold': '保持抓持', 'release_outside': '目标外释放'}
GOAL_LABEL_ZH = {'not_satisfied': '未满足', 'satisfied': '已满足', 'unknown': '不确定'}

ZH={'approach_object':'靠近物体','grasp':'夹取尝试','transport':'搬运 / 靠近托盘','unknown':'不确定',
    'unheld':'未抓持','held':'稳定抓持','not_placed':'未确认放置完成','placed':'已放置',
    'none':'无明显变化','drop':'掉落','regrasp':'重新抓持','release':'释放'}
HN={'phase':'阶段','holding':'抓持','placement':'放置','transition':'变化事件'}
COLORS={'unknown':(90,95,105),'approach_object':(55,145,230),'grasp':(235,175,55),'transport':(50,185,150),
        'held':(50,185,150),'unheld':(100,140,220),'not_placed':(100,140,220),'placed':(70,195,110),
        'none':(100,140,220),'release':(210,140,220),'drop':(230,75,70),'regrasp':(240,170,50)}
COLORS.update(supported=(70,195,110), unsupported=(230,120,80), inside=(70,195,110),
              outside=(230,120,80), satisfied=(70,195,110), not_satisfied=(100,140,220))


def transition_support(contract):
    counts = contract.get('observer_training_counts', {}).get('transition', [])
    missing = [name for name in ('drop', 'regrasp')
               if len(counts) <= HEADS['transition'].index(name) or counts[HEADS['transition'].index(name)] <= 0]
    message = ('缺正样本：' + ' / '.join(ZH[name] for name in missing)) if missing else '掉落 / 重抓：有训练样本'
    return missing, message + '；未独立测试'


def render(raw,predictions,detections,annotation_receipt,output,font):
    if output.exists():
        raise ValueError('refusing overwrite of rendered output')
    with np.load(predictions,allow_pickle=False) as archive:
        a={key:archive[key] for key in archive.files}
    contract=json.loads(str(a.pop('contract_json').item()))
    relational = bool(contract.get('relation_schema'))
    evidence_v4 = contract.get('format', '').startswith('umi_yolo_relational_evidence_gnn_v1')
    relation_names = list(contract.get('relation_schema', {}))
    goal_names = list(contract.get('goal_schema', {}))
    requested_goal = contract.get('requested_goal', 'place_on')
    missing_positive_classes, support_message = transition_support(contract)
    provenance=json.loads(annotation_receipt.read_text())
    det=[json.loads(line) for line in detections.read_text().splitlines() if line.strip()]
    lookup={(r['attempt_uid'],r['query_id']):r for r in det}
    if len(lookup)!=len(det) or set(lookup)!=set(zip(a['attempt_uid'].tolist(),a['query_id'].tolist())):
        raise ValueError('prediction/detection keys differ')
    output.mkdir(parents=True)
    font_path=str(font)
    big=ImageFont.truetype(font_path,25,index=2)
    small=ImageFont.truetype(font_path,18,index=2)
    title_font=ImageFont.truetype(font_path,28,index=2)
    outputs=[]
    for info in provenance['source_videos']:
        uid=info['attempt_uid']
        source=raw/info['file']
        if sha256(source)!=info['sha256']:
            raise ValueError('source video changed')
        indices=np.flatnonzero(a['attempt_uid']==uid)
        indices=indices[np.argsort(a['elapsed_s'][indices])]
        times=a['elapsed_s'][indices].tolist()
        stream=[]
        for i in indices:
            record=dict(attempt_uid=uid,elapsed_s=float(a['elapsed_s'][i]),query_id=int(a['query_id'][i]),
                        split=str(a['split'][i]),event_source='trained_gnn_causal_predictions_not_annotation',
                        yolo=lookup[(uid,int(a['query_id'][i]))]['detections']['umi'],events={},forecast={})
            for name,classes in HEADS.items():
                if f'{name}_id' not in a:
                    continue
                raw_id=int(a[f'{name}_id'][i]); label=classes[raw_id]
                counts=contract.get('observer_training_counts',{}).get(name,[])
                unsupported=bool(counts and counts[raw_id]==0)
                if unsupported:
                    label='unknown'
                record['events'][name]=dict(raw_prediction=label,raw_probabilities=a[f'{name}_probabilities'][i].tolist(),
                    confidence=float(a[f'{name}_probabilities'][i].max()),unsupported_class_suppressed=unsupported)
                if name!='transition':
                    record['events'][name].update(confirmed=classes[int(a[f'{name}_stable_id'][i])],
                                                 estimate=classes[int(a[f'{name}_belief_id'][i])])
                else:
                    record['events'][name].update(confirmed=label,estimate=label)
                if evidence_v4 and name in ('phase', 'holding'):
                    record['events'][name]['semantics'] = (
                        'relation_belief_temporal_inference_not_direct_observation'
                        if name == 'holding' else
                        'causal_temporal_event_state_not_direct_visual_evidence_confirmation')
            for k,horizon in enumerate(contract.get('forecast_horizons_s',[])):
                key=f'forecast_h{k}_placement_probabilities'
                p=a[key][i]
                record['forecast'][str(horizon)]=dict(placement=HEADS['placement'][int(p.argmax())],
                    probabilities=p.tolist(),is_current_fact=False)
            if relational:
                record['relations'] = {}
                for j, name in enumerate(relation_names):
                    classes = contract['relation_schema'][name]
                    probability = a[f'relation_{name}_probabilities'][i]
                    if evidence_v4:
                        current = a[f'current_relations_{name}_probabilities'][i]
                        prior = a[f'prior_relations_{name}_probabilities'][i]
                        record['relations'][name] = dict(
                            belief=dict(probabilities=probability.tolist(),
                                temporal_inference=classes[int(a['relation_stable_ids'][i,j])]),
                            current_evidence=dict(probabilities=current.tolist(),
                                prediction=classes[int(current.argmax())]),
                            prior_imagination=dict(probabilities=prior.tolist(), is_observed_fact=False),
                            observability=dict(probability=float(a['observability_probabilities'][i,j]),
                                is_calibrated=False),
                            source='graph_visual_gnn_causal_prediction_not_annotation')
                    else:
                        record['relations'][name] = dict(probabilities=probability.tolist(),
                            confirmed=classes[int(a['relation_stable_ids'][i,j])],
                            raw_prediction=classes[int(probability.argmax())],
                            source='graph_role_gnn_prediction_not_annotation')
                g = goal_names.index(requested_goal)
                if evidence_v4:
                    record['task_goal'] = dict(name=requested_goal, predicates=contract['goal_schema'][requested_goal],
                        inferred_state=contract['goal_classes'][int(a['goal_inferred_ids'][i,g])],
                        confirmed_state=contract['goal_classes'][int(a['goal_confirmed_ids'][i,g])],
                        state=contract['goal_classes'][int(a['goal_confirmed_ids'][i,g])],
                        probabilities=a['goal_probabilities'][i,g].tolist(), is_calibrated_value=False,
                        semantics='inference_and_evidence_confirmation_are_separate')
                else:
                    record['task_goal'] = dict(name=requested_goal, predicates=contract['goal_schema'][requested_goal],
                        state=contract['goal_classes'][int(a['goal_state_ids'][i,g])],
                        probabilities=a['goal_probabilities'][i,g].tolist(), is_calibrated_value=False)
                record['relation_forecast'] = {str(horizon): {
                    name: a[f'forecast_h{k}_{name}_probabilities'][i].tolist() for name in relation_names}
                    for k, horizon in enumerate(contract.get('relation_forecast_horizons_s', []))}
            stream.append(record)
        json_path=output/f"{source.stem}.predictions.jsonl"
        with json_path.open('x',encoding='utf-8') as f:
            for r in stream:
                f.write(json.dumps(r,ensure_ascii=False)+'\n')
        # Bottom area is separate, leaving the whole original field of view visible.
        width,height=1280,1280 if relational else 1040
        timeline_y=326 if evidence_v4 else 292 if relational else 180
        panel=Image.new('RGB',(width,height-720),(20,25,35))
        draw=ImageDraw.Draw(panel)
        heading=('对象关系证据 + Role-Graph Event Observer（预测非真值）' if evidence_v4 else
                 '对象关系图 GNN Event Observer（不是事件真值）') if relational else 'Event Observer + YOLO 辅助线索（不是事件真值）'
        draw.text((18,8),heading,font=title_font,fill='white')
        draw.text((18,timeline_y-30),'事件演变（离线全程展示，不作为模型输入）',font=small,fill=(190,195,205))
        duration=info['duration_s']
        for n,name in enumerate(HEADS):
            y=timeline_y+n*28
            draw.text((18,y-3),HN[name],font=small,fill='white')
            for j,r in enumerate(stream):
                stop=times[j+1] if j+1<len(times) else duration
                label=r['events'][name]['confirmed']
                x1=140+int(1120*times[j]/duration); x2=140+int(1120*stop/duration)
                draw.rectangle((x1,y,max(x1,x2),y+18),fill=COLORS[label])
        if relational:
            for n, name in enumerate(('supported_by_target', 'in_target_region', 'task_goal'), start=4):
                y=timeline_y+n*28
                draw.text((18,y-3),'任务满足' if name=='task_goal' else RELATION_ZH[name],font=small,fill='white')
                for j, r in enumerate(stream):
                    stop=times[j+1] if j+1<len(times) else duration
                    label=(r['task_goal']['state'] if name=='task_goal' else
                           r['relations'][name]['belief']['temporal_inference'] if evidence_v4 else
                           r['relations'][name]['confirmed'])
                    x1=140+int(1120*times[j]/duration); x2=140+int(1120*stop/duration)
                    draw.rectangle((x1,y,max(x1,x2),y+18),fill=COLORS[label])
        cv2.imwrite(str(output/f'{source.stem}.timeline.jpg'),cv2.cvtColor(np.asarray(panel),cv2.COLOR_RGB2BGR))
        cap=cv2.VideoCapture(str(source))
        target=output/f'{source.stem}.labeled.mp4'
        command=['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24',
                 '-s',f'{width}x{height}','-r','30','-i','pipe:0','-an','-c:v','libx264','-threads','2',
                 '-preset','veryfast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
        process=subprocess.Popen(command,stdin=subprocess.PIPE)
        frame=0
        try:
            while True:
                ok,bgr=cap.read()
                if not ok: break
                t=frame/30
                j=bisect.bisect_right(times,t+1e-7)-1
                if j<0: raise ValueError('no causal prediction at video start')
                r=stream[j]
                view=Image.fromarray(cv2.cvtColor(cv2.resize(bgr,(1280,720)),cv2.COLOR_BGR2RGB))
                d=ImageDraw.Draw(view)
                for box in r['yolo']:
                    color={'object':(65,240,90),'target':(70,170,255),'gripper':(255,100,90)}[box['role']]
                    xy=(np.asarray(box['xyxy'])*[1280,720,1280,720]).astype(int).tolist()
                    d.rectangle(xy,outline=color,width=3)
                    text={'object':'物体','target':'托盘','gripper':'夹持器'}[box['role']]+f" {box['confidence']:.2f}"
                    tx,ty=xy[0],max(0,xy[1]-26)
                    d.rectangle((tx,ty,min(1280,tx+180),ty+26),fill=(15,20,25))
                    d.text((tx,ty),text,font=small,fill=color)
                d.rectangle((0,0,1280,34),fill=(15,20,25))
                d.text((12,3),f"{source.stem} | {t:.2f}s | {info['split']} | YOLO/Role-GNN 10Hz，上次更新 {times[j]:.1f}s",font=small,fill='white')
                bottom=panel.copy(); d=ImageDraw.Draw(bottom)
                for k,name in enumerate(HEADS):
                    e=r['events'][name]; label=e['confirmed']
                    status=('时序推断' if evidence_v4 and name=='holding' and label!='unknown' else
                            '连续确认' if name!='transition' and label!='unknown' else
                            '当前预测' if name=='transition' else '未确认')
                    if label=='unknown' and e['estimate']!='unknown':
                        desc=f"历史推测：{ZH[e['estimate']]}（非确认）"
                    elif label=='unknown' and e['raw_prediction']!='unknown':
                        desc=f"待确认：{ZH[e['raw_prediction']]}"
                    else:
                        desc=f"{ZH[label]} / {status}"
                    x=18+(k%2)*630; y=48+(k//2)*40
                    d.text((x,y),f"{HN[name]}：{desc}",font=big,fill=COLORS.get(label,(220,220,220)) if label!='unknown' else (205,205,210))
                if relational:
                    g=r['task_goal']
                    if evidence_v4:
                        desc='   |   '.join(RELATION_ZH[name]+'推断：'+RELATION_LABEL_ZH[r['relations'][name]['belief']['temporal_inference']]
                            for name in relation_names)
                        evidence='   |   '.join(RELATION_ZH[name]+'证据可信度 '+f"{r['relations'][name]['observability']['probability']:.2f}"
                            for name in relation_names)
                        d.text((18,126),desc,font=small,fill=(220,225,230))
                        d.text((18,151),evidence+'（未校准）',font=small,fill=(190,200,215))
                        d.text((18,179),f"任务：{GOAL_ZH[g['name']]}  推断={GOAL_LABEL_ZH[g['inferred_state']]}  证据确认={GOAL_LABEL_ZH[g['confirmed_state']]}",
                               font=small,fill=COLORS[g['confirmed_state']])
                        d.text((18,204),f"网络成功分数 {g['probabilities'][1]:.2f}（非价值、非真值）",font=small,fill=(225,185,100))
                    else:
                        desc='   |   '.join(RELATION_ZH[name]+'：'+RELATION_LABEL_ZH[r['relations'][name]['confirmed']]
                                            for name in relation_names)
                        d.text((18,134),desc,font=small,fill=(220,225,230))
                        d.text((18,170),f"任务：{GOAL_ZH[g['name']]}  |  {GOAL_LABEL_ZH[g['state']]}  |  网络成功分数 {g['probabilities'][1]:.2f}（非价值）",font=big,fill=COLORS[g['state']])
                    future=r['relation_forecast'].get('0.6',{})
                    ft='；'.join(RELATION_ZH[name]+f" {future[name][1]:.2f}" for name in relation_names if name in future)
                    d.text((18,232 if evidence_v4 else 210),'0.6s 后关系预测：'+ft+'（不是当前事实）',font=small,fill=(225,185,100))
                    d.text((18,259 if evidence_v4 else 239),support_message+'；任务评价未验证跨任务/跨本体',font=small,fill=(225,185,100))
                else:
                    forecast=r['forecast'].get('0.6')
                    ft=f"0.6s 后放置概率 {forecast['probabilities'][1]:.2f}（预测≠当前完成）" if forecast else ''
                    d.text((18,125),f"{ft}   {support_message}",font=small,fill=(225,185,100))
                x=140+int(1120*t/duration)
                d.line((x,timeline_y-2,x,timeline_y+(193 if relational else 105)),fill='white',width=2)
                d.text((18,528 if relational else 294),'绿色框=物体  蓝色框=托盘  红色框=夹持器；没框是未检出，不代表没抓住。',font=small,fill=(180,190,205))
                combined=Image.new('RGB',(width,height));combined.paste(view,(0,0));combined.paste(bottom,(0,720))
                process.stdin.write(np.asarray(combined).tobytes())
                frame+=1
        finally:
            cap.release();process.stdin.close()
        if process.wait()!=0 or frame!=info['frames']:
            raise ValueError('video render failed or frame count changed')
        outputs.append(dict(source=source.name,video=target.name,predictions=json_path.name,frames=frame,
                            sha256=sha256(target),split=info['split']))
        print(json.dumps(outputs[-1]),flush=True)
    receipt=dict(format='eksf_umi_prediction_video_export_v1',event_source='trained_role_graph_event_observer_predictions',
        checkpoint_sha256=contract['checkpoint_sha256'],predictions_sha256=sha256(predictions),
        detections_sha256=sha256(detections),source_fps=30,model_and_detector_hz=10,
        display_alignment='causal_asof_no_future_boxes',training_labels_displayed=False,
        outputs=outputs,raw_videos_unchanged=True,human_review_required_to_generate=False,
        deployment_authorized=False,missing_positive_classes=missing_positive_classes)
    if relational:
        receipt.update(predictor_format=contract['format'], requested_goal=requested_goal,
                       relation_predictions_displayed=True, goal_score_is_calibrated_value=False)
        if evidence_v4:
            receipt.update(evidence_confirmation_separate_from_inference=True,
                           observability_is_calibrated=False)
    with (output/'export_receipt.json').open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
    with (output/'index.html').open('x',encoding='utf-8') as f:
        f.write('<!doctype html><meta charset="utf-8"><title>UMI 事件预测</title><h1>Role-Graph Event Observer＋YOLO 辅助框</h1><p>预测非真值；train 视频是训练集回放，不是独立验证。</p>')
        for r in outputs:
            f.write(f'<h3>{html.escape(r["source"])} ({r["split"]})</h3><video controls preload="none" width="800" src="{html.escape(r["video"])}"></video>')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('raw','predictions','detections','annotation-receipt','output','font'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(render(a.raw,a.predictions,a.detections,a.annotation_receipt,a.output,a.font),ensure_ascii=False))
