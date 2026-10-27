#!/usr/bin/env python3
"""Reproduce pre-gate attribution and calibrated sampled-peak upper bounds."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys

from study_utils import ROOT,dump


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('prepare','gate','allocation','all'),default='all')
    p.add_argument('--project-dir',type=Path,default=ROOT.parent)
    p.add_argument('--study-root',type=Path,default=ROOT.parent/'outputs/study')
    p.add_argument('--output-root',type=Path,default=ROOT.parent/'outputs/followup')
    p.add_argument('--data-csv',type=Path)
    p.add_argument('--split-manifest',type=Path)
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(40,45)))
    p.add_argument('--variants',nargs='+',choices=('full','no_code'),default=['full','no_code'])
    p.add_argument('--fit-fraction',type=float,default=.70)
    p.add_argument('--selection-end',type=float,default=.85)
    p.add_argument('--budgets',nargs='+',type=float,default=[0.,.25,.5,.75,1.])
    p.add_argument('--alphas',nargs='+',type=float,default=[.05,.01],help='Nominal joint job-level miscoverage; divide by 2 per resource')
    p.add_argument('--min-group-n',type=int,default=30)
    p.add_argument('--cache-capacity',type=int,default=1024)
    p.add_argument('--bootstrap-replicates',type=int,default=10000)
    p.add_argument('--bootstrap-seed',type=int,default=2026)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    if len(set(a.seeds))!=len(a.seeds) or len(set(a.variants))!=len(a.variants):
        p.error('Seeds and variants must be unique')
    if not .5<=a.fit_fraction<a.selection_end<1:
        p.error('Require .5 <= fit-fraction < selection-end < 1')
    if len(set(a.budgets))!=len(a.budgets) or any(not math.isfinite(b) or not 0<=b<=1 for b in a.budgets):
        p.error('Budgets must be unique finite numbers in [0,1]')
    if len(set(a.alphas))!=len(a.alphas) or any(not math.isfinite(x) or not 0<x<1 for x in a.alphas):
        p.error('Alphas must be unique finite numbers in (0,1)')
    if a.bootstrap_replicates<1 or a.min_group_n<1 or a.cache_capacity<0:
        p.error('Invalid bootstrap/group/cache size')
    if a.stage in ('allocation','all') and set(a.variants)!={'full','no_code'}:
        p.error('Allocation comparison requires both Full and No-Code')
    if a.stage in ('gate','all') and 'full' not in a.variants:
        p.error('The gate attribution and encoder-cost study requires Full')
    if a.stage in ('gate','all') and .5 not in a.budgets:
        p.error('The prespecified primary gate budget 0.5 must be included')
    return a


def environment():
    versions={}
    for name in ('numpy','pandas','scikit-learn','joblib','PyYAML'):
        try:
            versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name]=None
    return dict(python=sys.version,platform=platform.platform(),versions=versions,gpu_required=False)


def main(argv=None):
    a=parse_args(argv)
    plan=dict(stage=a.stage,seeds=a.seeds,variants=a.variants,study_root=str(a.study_root.resolve()),
        output_root=str(a.output_root.resolve()),base_model_retraining=False,gpu_required=False,
        lightweight_gate_refit=a.stage in ('prepare','all'),
        gate_attribution_variant='full only; No-Code has no code encoder',
        validation_roles=dict(gate_fit_fraction=a.fit_fraction,
            policy_selection_fraction=a.selection_end-a.fit_fraction,
            independent_calibration_fraction=1-a.selection_end,completion_purged=True),
        primary_gate_comparison='50% exact-history eligible bypass budget on validation; test realization reported',
        experiment_scope='separate adaptive-routing 70/15/15 comparative follow-up; not the primary exact-bypass 70/30 study',
        safety_scope='observed normalized sampled peaks, not real OOM or utilization gains')
    print(json.dumps(plan,indent=2),flush=True)
    if a.dry_run:
        return 0
    a.output_root=a.output_root.resolve()
    if a.stage in ('prepare','all'):
        from prepare_followup import prepare_one
        for seed in a.seeds:
            for variant in a.variants:
                prepare_one(a,seed,variant)
    if a.stage in ('gate','all'):
        from gate_study import gate_one,summarize_gate
        for seed in a.seeds:
            gate_one(a,seed,'full')
        summarize_gate(a)
    if a.stage in ('allocation','all'):
        from allocation_study import allocation_one,summarize_allocation
        for seed in a.seeds:
            allocation_one(a,seed)
        summarize_allocation(a)
    a.output_root.mkdir(parents=True,exist_ok=True)
    # Per-invocation receipt; completed stage manifests protect all actual results.
    from datetime import datetime,timezone
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    dump(a.output_root/f'invocation_{stamp}.json',dict(plan=plan,environment=environment()))
    print('[COMPLETE] No original experiment outputs were overwritten.',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
