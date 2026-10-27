"""Independent-calibration upper bounds and sampled-peak provisioning diagnostics.

These are normalized observed-target bounds, not measured OOM probabilities or
physical scheduler savings. Temporal drift invalidates an unconditional claim
of exchangeable split-conformal coverage; actual test coverage is always shown.
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
import pandas as pd

from study_utils import (RESOURCES,dump,publish_stage,rate_ci,read_prepared,
    save_summary,sha,source_inventory,verify_stage)


def conformal_margin(scores,alpha):
    """One-sided finite-sample order statistic, including the +infinity atom."""
    x=np.asarray(scores,float)
    if not 0<alpha<1 or not np.isfinite(x).all():
        raise ValueError('Invalid alpha or calibration residuals')
    k=math.ceil((len(x)+1)*(1-alpha)-1e-12)
    if not len(x) or k>len(x):
        return dict(n=len(x),rank=k,margin=None,full_bound=True,reason='finite_sample_rank_exceeds_calibration_size')
    q=max(0.,float(np.partition(x,k-1)[k-1]))
    return dict(n=len(x),rank=k,margin=q,full_bound=False,reason='finite_sample_upper_residual_quantile')


def apply_margin(prediction,record):
    p=np.asarray(prediction,float)
    if record['full_bound']:
        return np.ones_like(p)
    return np.clip(p+record['margin'],0.,1.)


def history_group(frame):
    return np.where(~frame.history_available.to_numpy(bool),'cold_start',
        np.where(frame.exact_script_history_available.to_numpy(bool),'exact_history','user_history_only'))


def fit_bound(frame,prediction,cal_mask,alpha,kind,min_group_n=30):
    """Fit with calibration rows only; alpha is per-resource, not joint."""
    p=np.asarray(prediction,float)
    target=frame.target.to_numpy(float)
    valid=frame.valid.to_numpy(bool)
    group=history_group(frame)
    upper=np.empty(len(frame),float)
    records=[]
    for resource in RESOURCES:
        rmask=frame.resource.eq(resource).to_numpy()
        fit=np.asarray(cal_mask,bool)&valid&rmask
        if not fit.any():
            raise ValueError(f'No valid {resource} calibration labels')
        if kind=='mondrian_history':
            for name in ('cold_start','exact_history','user_history_only'):
                train=fit&(group==name)
                rec=conformal_margin(target[train]-p[train],alpha)
                if int(train.sum())<min_group_n:
                    rec.update(full_bound=True,margin=None,reason='sparse_history_group_full_bound')
                use=rmask&(group==name)
                upper[use]=apply_margin(p[use],rec)
                records.append(dict(resource=resource,history_group=name,kind=kind,alpha_resource=alpha,**rec))
        else:
            residual=target[fit]-p[fit]
            if kind=='day_max':
                # Whole-day maximum scores address within-day multiplicity.
                # Exchangeability of day maxima is still an assumption, not a trace fact.
                days=frame.loc[fit,'submit_time'].dt.floor('D').astype(str)
                residual=pd.DataFrame({'day':days.to_numpy(),'score':residual}).groupby('day').score.max().to_numpy()
            elif kind!='global':
                raise ValueError(kind)
            rec=conformal_margin(residual,alpha)
            upper[rmask]=apply_margin(p[rmask],rec)
            records.append(dict(resource=resource,history_group='all',kind=kind,alpha_resource=alpha,**rec))
    if not np.isfinite(upper).all() or np.any(upper<p-1e-12):
        raise AssertionError('Upper bounds must be finite and no lower than point predictions')
    return upper,records


def alignment(full,other):
    keys=['sample_id','resource']
    if not full[keys].equals(other[keys]):
        raise ValueError('Full/No-Code row populations or order differ')
    for c in ('valid','resource_applicable','submit_time','completion_time','role'):
        if not full[c].equals(other[c]):
            raise ValueError(f'Full/No-Code {c} differs')
    if not np.allclose(full.target,other.target,rtol=0,atol=1e-12,equal_nan=True):
        raise ValueError('Full/No-Code labels differ')


def evaluate_bound(frame,point,upper,mask,*,seed,method,kind,alpha_job,replicates,bootstrap_seed):
    result=[]
    y=frame.target.to_numpy(float)
    valid=frame.valid.to_numpy(bool)
    group=history_group(frame)
    days=frame.submit_time.dt.floor('D').astype(str).to_numpy()
    for resource in RESOURCES:
        base=np.asarray(mask,bool)&valid&frame.resource.eq(resource).to_numpy()
        strata={'all':base,'high_utilization':base&(y>=.8),
            **{n:base&(group==n) for n in ('cold_start','exact_history','user_history_only')}}
        for subgroup,use in strata.items():
            if not use.any():
                continue
            exceed=y[use]>upper[use]+1e-12
            ci=rate_ci(exceed,days[use],replicates,bootstrap_seed,f'bound:{seed}:{method}:{kind}:{alpha_job}:{resource}:{subgroup}')
            result.append(dict(seed=seed,method=method,calibrator=kind,alpha_job=alpha_job,
                alpha_resource=alpha_job/2,resource=resource,subgroup=subgroup,
                n=ci['n'],day_clusters=ci['clusters'],shortfall_rate=ci['rate'],
                shortfall_n=ci['events'],shortfall_day_n=ci['event_clusters'],
                degenerate_bootstrap=ci['degenerate_bootstrap'],
                shortfall_ci_lower=ci['ci_lower'],shortfall_ci_upper=ci['ci_upper'],
                coverage=1-ci['rate'],mean_bound=float(upper[use].mean()),
                mean_point=float(point[use].mean()),mean_added_margin=float((upper[use]-point[use]).mean()),
                mean_shortfall=float(np.maximum(y[use]-upper[use],0).mean()),
                p95_shortfall=float(np.quantile(np.maximum(y[use]-upper[use],0),.95)),
                mean_excess_bound=float(np.maximum(upper[use]-y[use],0).mean()),
                full_bound_rate=float((upper[use]>=1-1e-12).mean()),
                empirical_resource_target_met=bool(exceed.mean()<=alpha_job/2),
                observations_are_sampled_peaks_not_oom_events=True))
    return result


def joint_metrics(frame,upper,mask,seed,method,kind,alpha_job,replicates,bootstrap_seed):
    d=frame.loc[mask,['sample_id','submit_time','resource_applicable','valid','target']].copy()
    d['upper']=np.asarray(upper)[mask]
    d['unknown']=d.resource_applicable & ~d.valid
    d['shortfall']=d.resource_applicable & d.valid & (d.target>d.upper+1e-12)
    jobs=d.groupby('sample_id').agg(unknown=('unknown','any'),shortfall=('shortfall','any'),submit_time=('submit_time','first'))
    complete=jobs.loc[~jobs.unknown]
    ci=rate_ci(complete.shortfall.to_numpy(float),complete.submit_time.dt.floor('D').astype(str).to_numpy(),
        replicates,bootstrap_seed,f'joint:{seed}:{method}:{kind}:{alpha_job}')
    return dict(seed=seed,method=method,calibrator=kind,alpha_job=alpha_job,n_jobs=len(jobs),
        complete_label_jobs=len(complete),excluded_unknown_jobs=int(jobs.unknown.sum()),
        joint_shortfall_rate=ci['rate'],joint_ci_lower=ci['ci_lower'],joint_ci_upper=ci['ci_upper'],
        joint_shortfall_n=ci['events'],joint_shortfall_day_n=ci['event_clusters'],
        degenerate_bootstrap=ci['degenerate_bootstrap'],
        empirical_joint_target_met=bool(ci['rate']<=alpha_job) if ci['rate'] is not None else None,
        day_clusters=ci['clusters'],formal_temporal_coverage_guarantee=False)


def allocation_one(args,seed):
    root=Path(args.output_root)/'prepared'
    full_dir=root/'full'/f'seed_{seed}'
    no_dir=root/'no_code'/f'seed_{seed}'
    verify_stage(full_dir);verify_stage(no_dir)
    full=read_prepared(full_dir/'predictions.csv')
    no=read_prepared(no_dir/'predictions.csv')
    alignment(full,no)
    request=dict(stage='allocation',seed=seed,input_sha256={str(p):sha(p) for p in (full_dir/'predictions.csv',no_dir/'predictions.csv')},
        source_sha256=source_inventory(),alphas=args.alphas,min_group_n=args.min_group_n,
        bootstrap_replicates=args.bootstrap_replicates,bootstrap_seed=args.bootstrap_seed,
        calibration_role='independent validation tail after completion purge; boundaries in prepared protocol.json',
        alpha_split='alpha_job/2 for each resource',formal_temporal_coverage_claim=False)

    def build(out):
        methods={'full_cascade':full.prediction.to_numpy(float),
            'no_code_cascade':no.prediction.to_numpy(float),
            'hierarchical_last1':full.hierarchical_reference_prediction.to_numpy(float),
            'qwen_standalone':full.qwen_prediction.to_numpy(float)}
        cal=full.role.eq('calibration').to_numpy()
        test=full.role.eq('test').to_numpy()
        resources=[];joint=[];params=[]
        saved=full.loc[test,['sample_id','resource','valid','resource_applicable','target','submit_time']].copy()
        for method,point in methods.items():
            for alpha in args.alphas:
                for kind in ('global','mondrian_history','day_max'):
                    upper,records=fit_bound(full,point,cal,float(alpha)/2,kind,args.min_group_n)
                    params.extend(dict(seed=seed,method=method,alpha_job=alpha,**r) for r in records)
                    resources.extend(evaluate_bound(full,point,upper,test,seed=seed,method=method,kind=kind,
                        alpha_job=alpha,replicates=args.bootstrap_replicates,bootstrap_seed=args.bootstrap_seed))
                    joint.append(joint_metrics(full,upper,test,seed,method,kind,alpha,args.bootstrap_replicates,args.bootstrap_seed))
                    saved[f'{method}__{kind}__alpha_{alpha:g}']=upper[test]
                # Honest reference anchors: no calibration and a unit bound.
                for kind,upper in [('point_only',point),('unit_bound',np.ones(len(full)))]:
                    resources.extend(evaluate_bound(full,point,upper,test,seed=seed,method=method,kind=kind,
                        alpha_job=alpha,replicates=args.bootstrap_replicates,bootstrap_seed=args.bootstrap_seed))
                    joint.append(joint_metrics(full,upper,test,seed,method,kind,alpha,args.bootstrap_replicates,args.bootstrap_seed))
        pd.DataFrame(resources).to_csv(out/'resource_bounds_metrics.csv',index=False)
        pd.DataFrame(joint).to_csv(out/'joint_bounds_metrics.csv',index=False)
        pd.DataFrame(params).to_csv(out/'calibration_parameters.csv',index=False)
        saved.to_csv(out/'PRIVATE_test_bounds.csv',index=False)
        dump(out/'scope.json',dict(new_base_training=False,model_selection_on_calibration=False,
            test_labels_used_for_calibration_or_policy_selection=False,
            bound_target='observed normalized busiest-node / busiest-GPU sampled peak',
            nominal_joint_miscoverage_targets=args.alphas,resource_alpha='joint target divided by 2',
            finite_sample_quantile='ceil((n+1)*(1-alpha_resource)); +infinity atom maps to unit bound',
            sparse_mondrian_groups='unit bound; no optimistic global fallback',
            day_max='calibrate one score per calibration day: maximum resource residual',
            exchangeability='Conditional theoretical premise only; temporal production jobs and days are not assumed proven exchangeable.',
            no_claims=['real OOM probability','unobserved spikes','safe co-location','physical byte savings','real utilization gains'],
            cost_measure='mean normalized upper bound and added margin; not summed physical capacities',
            all_seeds_share_one_test_period=True,
            zero_event_interval='Percentile bootstrap can degenerate to [0,0]; this is not evidence of zero future risk.',
            bootstrap='submission-day cluster percentile intervals for individual rates; fixed bounds; pointwise'))
    publish_stage(Path(args.output_root)/'allocation'/f'seed_{seed}',request,build,args.resume)


def summarize_allocation(args):
    root=Path(args.output_root)/'allocation'
    for file,groups in (
        ('resource_bounds_metrics.csv',['method','calibrator','alpha_job','resource','subgroup']),
        ('joint_bounds_metrics.csv',['method','calibrator','alpha_job'])):
        paths=[root/f'seed_{s}'/file for s in args.seeds]
        if not all(p.is_file() for p in paths):
            continue
        frame=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
        frame.to_csv(root/file.replace('.csv','_per_seed.csv'),index=False)
        save_summary(frame,root/file.replace('.csv','_summary.csv'),groups)
