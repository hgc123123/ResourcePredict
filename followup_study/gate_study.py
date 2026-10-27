"""Budget-controlled, frozen-counterfactual tests of pre-inference routing."""
from __future__ import annotations
from collections import OrderedDict
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

from study_utils import (RESOURCES,cluster_difference_ci,dump,metrics,publish_stage,
    read_prepared,save_summary,sha,source_inventory,verify_stage)

STRATEGIES=('learned','recent_history','low_dispersion','many_observations','random_hash')


def tie_uniform(frame,seed):
    return np.array([int.from_bytes(hashlib.sha256(f'{seed}:{s}:{r}'.encode()).digest()[:8],'big')/2**64
        for s,r in zip(frame.sample_id,frame.resource)])


def score(frame,strategy):
    if strategy=='learned':
        x=frame.reliability_probability.to_numpy(float)
    elif strategy=='recent_history':
        x=-np.log1p(np.maximum(frame.exact_history_age_hours.to_numpy(float),0))
    elif strategy=='low_dispersion':
        x=-frame.exact_history_mad5.to_numpy(float)
    elif strategy=='many_observations':
        x=np.log1p(frame.exact_history_count.to_numpy(float))
    elif strategy=='random_hash':
        x=np.zeros(len(frame))
    else:
        raise ValueError(strategy)
    # Missing history statistics rank last; use a serializable finite sentinel.
    return np.where(np.isfinite(x),x,-np.finfo(float).max)


def eligible(frame):
    return (frame.resource_applicable & frame.history_available & frame.exact_script_history_available).to_numpy(bool)


def fit_cutoff(scores,ties,budget):
    """Freeze a scalar score cutoff and target-independent tie cutoff on selection."""
    s,u=np.asarray(scores,float),np.asarray(ties,float)
    if not 0<=budget<=1:
        raise ValueError('Budget must be in [0,1]')
    if not len(s):
        raise ValueError('No eligible selection rows')
    k=int(math.floor(len(s)*budget+1e-12))
    if k==0:
        return dict(mode='none',selection_eligible=len(s),selection_bypass=0,budget=budget)
    if k==len(s):
        return dict(mode='all',selection_eligible=len(s),selection_bypass=k,budget=budget)
    order=np.lexsort((-u,-s))
    # The last included pair is fixed; no test-period quantile or future job is used.
    j=order[k-1]
    return dict(mode='cutoff',score=float(s[j]),tie=float(u[j]),selection_eligible=len(s),selection_bypass=k,budget=budget)


def apply_cutoff(scores,ties,cutoff):
    if cutoff['mode']=='none':
        return np.zeros(len(scores),bool)
    if cutoff['mode']=='all':
        return np.ones(len(scores),bool)
    return (scores>cutoff['score'])|((scores==cutoff['score'])&(ties>=cutoff['tie']))


def routed_prediction(frame,bypass):
    history=frame.history_expert_prediction.to_numpy(float)
    post=frame.post_blend_prediction.to_numpy(float)
    if not np.isfinite(post).all() or not np.isfinite(history[bypass]).all():
        raise ValueError('Counterfactual post/history predictions are incomplete')
    return np.clip(np.where(bypass,history,post),0,1)


def cache_counts(frame,bypass,capacity=1024):
    """Empty cache after policy freeze; calibration prefix; job-level OR."""
    # Final policy and score cutoffs do not exist during gate-fit/selection.
    # Calibration-period labels may update causal history after completion,
    # but are never used to change these frozen routing decisions.
    include=frame.role.isin(('calibration','calibration_purged','test')).to_numpy()
    f=frame.loc[include].copy()
    f['required']=f.resource_applicable.to_numpy(bool)&~np.asarray(bypass,bool)[include]
    jobs=f.groupby('sample_id',sort=False).agg(submit_time=('submit_time','first'),split=('split','first'),
        required=('required','any'),script_sha256=('script_sha256','first'),script_observed=('script_observed','first'))
    jobs=jobs.reset_index().sort_values(['submit_time','sample_id'],kind='stable')
    cache=OrderedDict(); records=[]
    for _,group in jobs.groupby('submit_time',sort=False):
        pending=OrderedDict()
        for row in group.itertuples(index=False):
            key=str(row.script_sha256)
            hit=bool(capacity and row.required and row.script_observed and key in cache)
            if hit:
                cache.move_to_end(key)
            miss=bool(row.required and not hit)
            if capacity and miss and row.script_observed:
                pending[key]=None;pending.move_to_end(key)
            if row.split=='test':
                records.append(dict(sample_id=row.sample_id,qwen_required_job=bool(row.required),cache_hit=hit,encoder_forward=miss))
        for key in pending:
            cache[key]=None;cache.move_to_end(key)
            while len(cache)>capacity:
                cache.popitem(last=False)
    return pd.DataFrame(records)


def gate_one(args,seed,variant='full'):
    if variant!='full':
        raise ValueError('Encoder-cost gate attribution uses Full only; No-Code has no encoder')
    prepared=Path(args.output_root)/'prepared'/variant/f'seed_{seed}'
    verify_stage(prepared)
    frame=read_prepared(prepared/'predictions.csv')
    request=dict(stage='gate',seed=seed,variant=variant,input_sha256=sha(prepared/'predictions.csv'),
        source_sha256=source_inventory(),budgets=args.budgets,bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,cache_capacity=args.cache_capacity,
        cache_start='calibration after policy freeze; initially empty',
        thresholds_selected_on='policy_select only',primary_budget=.5,
        same_expert_post_model_trust_region_for_all_routers=True)

    def build(out):
        cutoff_records=[]; rows=[]; paired=[]; jobs_rows=[]; predictions={}
        ties=tie_uniform(frame,seed)
        elig=eligible(frame)
        test=frame.role.eq('test').to_numpy()
        selection=frame.role.eq('policy_select').to_numpy()
        for budget in args.budgets:
            for strategy in STRATEGIES:
                values=score(frame,strategy)
                bypass=np.zeros(len(frame),bool)
                for resource in RESOURCES:
                    resource_mask=frame.resource.eq(resource).to_numpy()
                    s=selection & resource_mask & elig
                    cutoff=fit_cutoff(values[s],ties[s],float(budget))
                    use=resource_mask & elig
                    bypass[use]=apply_cutoff(values[use],ties[use],cutoff)
                    cutoff_records.append(dict(seed=seed,resource=resource,strategy=strategy,**cutoff))
                pred=routed_prediction(frame,bypass)
                predictions[(strategy,float(budget))]=(pred,bypass)
        for name,bypass in [('no_bypass',np.zeros(len(frame),bool)),('exact_bypass',elig)]:
            predictions[(name,-1.)]=(routed_prediction(frame,bypass),bypass)
        for (strategy,budget),(pred,bypass) in predictions.items():
            counts=cache_counts(frame,bypass,args.cache_capacity)
            jobs_rows.append(dict(seed=seed,variant=variant,strategy=strategy,budget=budget,
                n_jobs=len(counts),qwen_required_job_rate=float(counts.qwen_required_job.mean()),
                encoder_forward_job_rate=float(counts.encoder_forward.mean()),cache_hit_n=int(counts.cache_hit.sum())))
            for resource in RESOURCES:
                mask=test & frame.valid.to_numpy(bool) & frame.resource.eq(resource).to_numpy()
                y=frame.target.to_numpy(float)[mask]
                exact_mask=mask & elig
                # Regret is relative to the identical frozen post-fusion path, not Qwen standalone.
                chosen=mask & bypass
                h_err=np.abs(frame.history_expert_prediction.to_numpy(float)[chosen]-frame.target.to_numpy(float)[chosen])
                p_err=np.abs(frame.post_blend_prediction.to_numpy(float)[chosen]-frame.target.to_numpy(float)[chosen])
                rows.append(dict(seed=seed,variant=variant,resource=resource,strategy=strategy,budget=budget,
                    **metrics(y,pred[mask]),bypass_rate=float(bypass[mask].mean()),
                    eligible_n=int(exact_mask.sum()),bypassed_n=int(chosen.sum()),
                    bypass_nonworse_than_post_rate=float((h_err<=p_err+1e-12).mean()) if len(h_err) else np.nan,
                    bypass_mae_regret_vs_post=float((h_err-p_err).mean()) if len(h_err) else np.nan,
                    test_eligible_bypass_fraction=float(bypass[exact_mask].mean()) if exact_mask.any() else np.nan))
        for budget in args.budgets:
            learned=predictions[('learned',float(budget))][0]
            for reference in (*STRATEGIES[1:],'no_bypass','exact_bypass'):
                ref=predictions[(reference,float(budget) if reference in STRATEGIES else -1.)][0]
                for resource in RESOURCES:
                    mask=test & frame.valid.to_numpy(bool) & frame.resource.eq(resource).to_numpy()
                    for result in cluster_difference_ci(frame.target.to_numpy(float)[mask],learned[mask],ref[mask],
                        frame.submit_time.dt.floor('D').astype(str).to_numpy()[mask],
                        replicates=args.bootstrap_replicates,seed=args.bootstrap_seed,label=f'gate:{seed}:{variant}:{resource}:{budget}:{reference}'):
                        paired.append(dict(seed=seed,variant=variant,resource=resource,budget=budget,
                            proposed='learned',reference=reference,**result))
        pd.DataFrame(rows).to_csv(out/'gate_metrics.csv',index=False)
        pd.DataFrame(jobs_rows).to_csv(out/'gate_job_costs.csv',index=False)
        pd.DataFrame(paired).to_csv(out/'gate_paired_intervals.csv',index=False)
        dump(out/'validation_frozen_cutoffs.json',cutoff_records)
        dump(out/'interpretation.json',dict(primary_comparison_budget=.5,
            budget_denominator='eligible exact-history resource rows in policy-selection block',
            budgets_matched_on_validation_not_guaranteed_matched_on_test=True,
            no_test_quantiles_or_labels_for_routing=True,
            independent_learning_benefit_requires='Better risk/error versus simple score routers with comparable realized job-level encoder demand; primary all-exact and no-bypass anchors must also be shown.',
            compute_counts_are_replay_not_measured_latency=True,
            cache_warmup='calibration submissions only, after policy freeze; no training or selection prewarming',
            component_selection='Shared history expert selected with learned-policy candidates; conditional router attribution, not independently optimized pipelines.',
            bootstrap='paired UTC submission-day; fixed predictions; pointwise, no multiplicity adjustment',
            no_positive_result_guaranteed=True))
    publish_stage(Path(args.output_root)/'gate'/variant/f'seed_{seed}',request,build,args.resume)


def summarize_gate(args):
    root=Path(args.output_root)/'gate'
    for file,groups in (
        ('gate_metrics.csv',['variant','resource','strategy','budget']),
        ('gate_job_costs.csv',['variant','strategy','budget']),
        ('gate_paired_intervals.csv',['variant','resource','budget','reference','metric'])):
        paths=[root/'full'/f'seed_{s}'/file for s in args.seeds]
        if not all(p.is_file() for p in paths):
            continue
        f=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
        f.to_csv(root/file.replace('.csv','_per_seed.csv'),index=False)
        if file!='gate_paired_intervals.csv':
            save_summary(f,root/file.replace('.csv','_summary.csv'),groups)
