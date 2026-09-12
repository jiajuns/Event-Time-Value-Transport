"""Causal confirmation and future supervision. Beliefs are never reward facts."""
from collections import deque
import numpy as np
from .factorized_events import HEADS


def future_label_targets(arrays, horizons, tolerance_s=.051, max_gap_s=.151):
    """Labels at t+h are supervision only; no future image/action is returned."""
    if any(not np.isfinite(h) or h <= 0 for h in horizons):
        raise ValueError('future horizons must be finite and positive')
    target = np.full((len(arrays['elapsed_s']),len(horizons),len(HEADS)),-1,np.int64)
    for uid in np.unique(arrays['attempt_uid']):
        indices = np.flatnonzero(arrays['attempt_uid']==uid)
        indices = indices[np.argsort(arrays['elapsed_s'][indices])]
        if len(set(arrays['split'][indices]))!=1:
            raise ValueError('attempt crosses splits')
        times=arrays['elapsed_s'][indices]
        for p,index in enumerate(indices):
            for k,horizon in enumerate(horizons):
                q=int(np.searchsorted(times,times[p]+horizon-1e-7))
                if q>=len(times) or abs(times[q]-times[p]-horizon)>tolerance_s:
                    continue
                if np.any(np.diff(times[p:q+1])>max_gap_s):
                    continue
                target[index,k]=arrays['labels'][indices[q]]
    return target


class CausalEventStabilizer:
    """Three observations incl. current; short occlusions yield non-confirmed beliefs.

    Do not smooth short transition events away. No camera visibility heuristic,
    monotonically increasing phase prior, or future forecast can confirm a fact.
    """
    def __init__(self, frames=3, confidence=.6, min_span_s=.18, max_gap_s=.151, belief_ttl_s=.2):
        if frames<2 or not 0<=confidence<=1 or not np.isfinite([min_span_s,max_gap_s,belief_ttl_s]).all() or min(min_span_s,max_gap_s,belief_ttl_s)<0:
            raise ValueError('invalid confirmation configuration')
        self.frames,self.confidence,self.min_span_s=frames,confidence,min_span_s
        self.max_gap_s,self.belief_ttl_s=max_gap_s,belief_ttl_s
        self.reset()

    def reset(self):
        self.history={h:deque(maxlen=self.frames) for h in HEADS if h!='transition'}
        self.last={}
        self.time=None

    def update(self, timestamp, probabilities):
        if not np.isfinite(timestamp) or (self.time is not None and timestamp<=self.time):
            raise ValueError('strictly increasing finite timestamps required')
        if self.time is not None and timestamp-self.time>self.max_gap_s:
            self.reset()
        self.time=timestamp
        raw={}
        for h,classes in HEADS.items():
            if h not in probabilities:
                continue
            p=np.asarray(probabilities[h],float)
            if p.shape!=(len(classes),) or not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1,atol=1e-4):
                raise ValueError('invalid probabilities')
            label=classes[int(p.argmax())]
            raw[h]=label if p.max()>=self.confidence else 'unknown'
        conflict=raw.get('holding')=='held' and raw.get('placement')=='placed'
        if conflict:
            raw['holding']=raw['placement']='unknown'
            for h in ('holding','placement'):
                self.last.pop(h,None)
                self.history[h].clear()
        if raw.get('transition') in ('drop','release'):
            # A brief observed loss/release of grasp overrides retained holding.
            self.last.pop('holding',None)
            self.history['holding'].clear()
            if raw.get('holding')=='held':
                raw['holding']='unknown'
            if raw['transition']=='drop':
                self.last.pop('placement',None)
                self.history['placement'].clear()
                raw['placement']='unknown'
        result={}
        for h,history in self.history.items():
            label=raw.get(h,'unknown')
            if label=='unknown':
                history.clear()
            else:
                if history and history[-1][1]!=label:
                    history.clear()
                if h in self.last and self.last[h][1]!=label:
                    self.last.pop(h)
                history.append((timestamp,label))
            confirmed=(len(history)==self.frames and timestamp-history[0][0]>=self.min_span_s-1e-7)
            if confirmed:
                self.last[h]=(timestamp,label)
            prior=self.last.get(h)
            retained=(label=='unknown' and prior is not None and timestamp-prior[0]<=self.belief_ttl_s+1e-7)
            result[h]=dict(confirmed=bool(confirmed),state=label if confirmed else 'unknown',
                           estimate=label if confirmed else prior[1] if retained else 'unknown',
                           source='three_frame_evidence' if confirmed else 'short_history_belief' if retained else 'uncertain')
        if result['placement']['state']=='placed' and result['holding']['state']!='unheld':
            result['placement']=dict(confirmed=False,state='unknown',estimate='unknown',source='needs_unheld_confirmation')
            self.last.pop('placement',None)
        result['transition']=dict(state=raw.get('transition','unknown'),source='current_unsmoothed_transition')
        result['holding_placement_conflict']=conflict
        return result
