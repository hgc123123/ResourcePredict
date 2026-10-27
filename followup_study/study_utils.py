"""Shared, dependency-light utilities for the two follow-up studies."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import sys
from uuid import uuid4

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESOURCES = ('memory', 'vram')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(x) for x in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def dump(path, payload):
    Path(path).write_text(json.dumps(clean(payload), indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def source_inventory(project=None):
    paths = sorted(p for p in ROOT.glob('*.py'))
    paths += sorted((ROOT/'scripts').glob('*')) if (ROOT/'scripts').is_dir() else []
    result = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths if p.is_file()}
    if project:
        for folder in ('reliability_cascade', 'common', 'qwen', 'dynamic_hybrid'):
            for p in sorted((Path(project)/folder).glob('*.py')):
                result['parent_project/'+p.relative_to(project).as_posix()] = sha(p)
    return result


def boolean(series):
    s = series.astype(str).str.lower().str.strip()
    if not s.isin(('true','false','1','0')).all():
        raise ValueError('Invalid or missing boolean field')
    return s.isin(('true','1'))


def read_prepared(path):
    p = Path(path)
    d = pd.read_csv(p, dtype={'sample_id':str,'resource':str,'user_id':str,'script_sha256':str})
    for col in ('valid','resource_applicable','exact_script_history_available','history_available','script_observed'):
        d[col] = boolean(d[col])
    for col in ('submit_time','completion_time'):
        d[col] = pd.to_datetime(d[col], utc=True, errors='raise')
        if d[col].isna().any():
            raise ValueError(f'Missing {col}')
    if d.duplicated(['sample_id','resource']).any():
        raise ValueError('Duplicate sample/resource rows')
    if set(d.resource) != set(RESOURCES):
        raise ValueError('Expected both resource rows for every job')
    if not d.groupby('sample_id').resource.nunique().eq(2).all():
        raise ValueError('Incomplete two-resource job population')
    valid = d.valid.to_numpy()
    for c in ('target','prediction','qwen_prediction','post_blend_prediction','hierarchical_reference_prediction'):
        v = pd.to_numeric(d.loc[valid,c],errors='raise').to_numpy(float)
        if not np.isfinite(v).all() or np.any((v < 0)|(v > 1)):
            raise ValueError(f'Invalid normalized {c}')
    return d.sort_values(['submit_time','sample_id','resource'],kind='stable').reset_index(drop=True)


def three_way_roles(frame, fit_fraction=.70, selection_end=.85):
    """Split by submission group, then purge labels unavailable at the next role."""
    if not .5 <= fit_fraction < selection_end < 1:
        raise ValueError('Require .5 <= fit_fraction < selection_end < 1')
    d = frame.copy()
    d['submit_time'] = pd.to_datetime(d.submit_time,utc=True,errors='raise')
    d['completion_time'] = pd.to_datetime(d.completion_time,utc=True,errors='raise')
    # Each job must have one immutable time/split record across resources.
    for c in ('submit_time','completion_time','split'):
        if not d.groupby('sample_id')[c].nunique(dropna=False).eq(1).all():
            raise ValueError(f'Inconsistent paired {c}')
    jobs = d.drop_duplicates('sample_id').sort_values(['submit_time','sample_id'],kind='stable')
    val = jobs.loc[jobs.split.eq('validation')]
    test = jobs.loc[jobs.split.eq('test')]
    if len(val) < 10 or test.empty:
        raise ValueError('Need nonempty validation and test populations')
    a = val.iloc[int(len(val)*fit_fraction)].submit_time
    b = val.iloc[int(len(val)*selection_end)].submit_time
    t = test.submit_time.min()
    if not a < b < t:
        raise ValueError('Submission ties or ordering create empty/overlapping roles')
    roles = pd.Series('excluded',index=d.index,dtype=object)
    validation = d.split.eq('validation')
    for name,mask,boundary in (
        ('gate_fit', validation & (d.submit_time < a),a),
        ('policy_select',validation & (d.submit_time >= a)&(d.submit_time < b),b),
        ('calibration',validation & (d.submit_time >= b),t)):
        roles.loc[mask] = name+'_purged'
        roles.loc[mask & (d.completion_time < boundary)] = name
    roles.loc[d.split.eq('test')] = 'test'
    if any(not roles.eq(n).any() for n in ('gate_fit','policy_select','calibration','test')):
        raise ValueError('Completion purging creates an empty role')
    counts = d.assign(role=roles).drop_duplicates('sample_id').groupby('role').size().to_dict()
    return roles, dict(fit_fraction=fit_fraction,selection_end_fraction=selection_end,
        selection_start=a.isoformat(),calibration_start=b.isoformat(),test_start=t.isoformat(),
        unique_jobs=counts,equal_submit_times_split=False,
        fit_selection_calibration_test_disjoint=True,
        calibration_labels_used_for_gate_fit_or_policy_selection=False)


def named_rng(seed,*parts):
    h=hashlib.sha256(json.dumps([int(seed),*parts],sort_keys=True).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:16],'big'))


def cluster_difference_ci(target, proposed, reference, clusters, *, replicates=10000, seed=2026, label=''):
    """Fixed-prediction paired day-cluster percentile intervals."""
    y,p,r = map(lambda x:np.asarray(x,float),(target,proposed,reference))
    if not len(y) or len(y)!=len(p) or len(y)!=len(r) or not all(np.isfinite(x).all() for x in (y,p,r)):
        raise ValueError('Paired bootstrap needs aligned finite predictions')
    labels,inv=np.unique(np.asarray(clusters,str),return_inverse=True)
    c=len(labels)
    n=np.bincount(inv,minlength=c)
    weights=named_rng(seed,label).multinomial(c,np.full(c,1/c),size=replicates)
    den=weights@n
    out=[]
    for name,loss,transform,favorable in (
        ('mae',lambda x:np.abs(x-y),lambda x:x,'negative'),
        ('rmse',lambda x:(x-y)**2,np.sqrt,'negative'),
        ('acc_at_10',lambda x:(np.abs(x-y)<=.10).astype(float),lambda x:x,'positive')):
        lp,lr=loss(p),loss(r)
        sp=np.bincount(inv,weights=lp,minlength=c)
        sr=np.bincount(inv,weights=lr,minlength=c)
        delta=transform(weights@sp/den)-transform(weights@sr/den)
        lo,hi=np.quantile(delta,[.025,.975])
        point=float(transform(lp.mean())-transform(lr.mean()))
        out.append(dict(metric=name,n=len(y),clusters=c,delta=point,ci_lower=float(lo),ci_upper=float(hi),
            favorable_interval=bool(hi<0 if favorable=='negative' else lo>0),
            bootstrap_replicates=replicates,pointwise=True,multiplicity_adjusted=False))
    return out


def rate_ci(values,clusters,replicates=10000,seed=2026,label=''):
    x=np.asarray(values,float)
    if not len(x):
        return dict(n=0,rate=None,ci_lower=None,ci_upper=None,clusters=0,
            events=0,event_clusters=0,degenerate_bootstrap=True)
    labs,inv=np.unique(np.asarray(clusters,str),return_inverse=True)
    n=np.bincount(inv); sums=np.bincount(inv,weights=x)
    w=named_rng(seed,label).multinomial(len(labs),np.full(len(labs),1/len(labs)),size=replicates)
    draws=(w@sums)/(w@n)
    lo,hi=np.quantile(draws,[.025,.975])
    return dict(n=len(x),rate=float(x.mean()),ci_lower=float(lo),ci_upper=float(hi),clusters=len(labs),
        events=int(np.count_nonzero(x)),event_clusters=int(np.count_nonzero(sums)),
        degenerate_bootstrap=bool(np.isclose(lo,hi,rtol=0,atol=1e-15)))


def metrics(y,p):
    y,p=np.asarray(y,float),np.asarray(p,float)
    e=p-y
    return dict(n=len(y),mae=float(np.abs(e).mean()),rmse=float(np.sqrt(np.mean(e*e))),acc_at_10=float((np.abs(e)<=.10).mean()))


def publish_stage(destination,request,builder,resume=False):
    """Only publish a complete stage; refuse changed-input or corrupt resumes."""
    destination=Path(destination)
    canonical=json.dumps(clean(request),sort_keys=True,separators=(',',':'),allow_nan=False)
    request_sha=hashlib.sha256(canonical.encode()).hexdigest()
    if destination.exists():
        record=destination/'completion.json'
        if not resume or not record.is_file():
            raise FileExistsError(f'{destination} exists; use --resume for a verified completed stage, or a new output root')
        m=json.loads(record.read_text(encoding='utf-8'))
        if m['request_sha256']!=request_sha:
            raise ValueError(f'Resume contract changed at {destination}; choose a new output root')
        for name,digest in m['files'].items():
            p=(destination/name).resolve()
            if not p.is_relative_to(destination.resolve()) or sha(p)!=digest:
                raise ValueError(f'Corrupt stage output: {name}')
        print(f'[SKIP verified] {destination}',flush=True)
        return
    destination.parent.mkdir(parents=True,exist_ok=True)
    temp=destination.parent/f'.{destination.name}.staging_{uuid4().hex}'
    temp.mkdir()
    dump(temp/'request.json',request)
    builder(temp)
    files={p.relative_to(temp).as_posix():sha(p) for p in temp.rglob('*') if p.is_file()}
    dump(temp/'completion.json',dict(request_sha256=request_sha,files=files,completed=True))
    temp.rename(destination)
    print(f'[DONE] {destination}',flush=True)


def verify_stage(path):
    path=Path(path)
    record=json.loads((path/'completion.json').read_text(encoding='utf-8'))
    for name,digest in record['files'].items():
        p=(path/name).resolve()
        if not p.is_relative_to(path.resolve()) or sha(p)!=digest:
            raise ValueError(f'Prepared stage integrity failure: {name}')
    return record


def summarize(frame,groups):
    columns=[c for c in frame.select_dtypes(include=[np.number]).columns if c not in set(groups)|{'seed'}]
    return frame.groupby(groups,dropna=False)[columns].agg(['mean','std','count']).reset_index()


def save_summary(frame,path,groups):
    s=summarize(frame,groups)
    s.columns=['_'.join(str(x) for x in c if x) if isinstance(c,tuple) else c for c in s.columns]
    s.to_csv(path,index=False)
