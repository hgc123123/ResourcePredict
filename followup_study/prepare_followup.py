"""CPU-only refitting of small gates with a separate calibration tail."""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from study_utils import dump, sha, publish_stage, source_inventory, three_way_roles


def prepare_one(args,seed,variant):
    project=Path(args.project_dir).resolve()
    if not (project/'reliability_cascade/pipeline.py').is_file():
        raise FileNotFoundError(f'Parent experiment project not found: {project}')
    sys.path.insert(0,str(project))
    try:
        from common.config import load_config, set_value, resolved_serializable_config
        from reliability_cascade.pipeline import prepare_cascade,cascade_model_config
        from reliability_cascade.selection import select_cascade
        from reliability_cascade.evaluation import apply_selected_cascade
    except ModuleNotFoundError as e:
        raise RuntimeError('Install requirements.txt in the cluster Python environment; preparation needs scikit-learn, joblib and PyYAML but no GPU or model weights.') from e
    study=Path(args.study_root).resolve()
    base=study/f'train_tail/base/qwen/tokens_4096/{variant}/seed_{seed}'
    last=study/f'train_tail/base/last-1/completed_before_submission/seed_{seed}'
    for p in (base/'config.resolved.yaml',base/'predictions.csv',base/'run_manifest.json',last/'predictions.csv',last/'run_manifest.json'):
        if not p.is_file():
            raise FileNotFoundError(p)
    config=load_config(base/'config.resolved.yaml')
    set_value(config,'seed',seed)
    set_value(config,'qwen.max_tokens',4096)
    set_value(config,'reliability_cascade.qwen_variant',variant)
    set_value(config,'reliability_cascade.require_independent_base_selection',True)
    set_value(config,'reliability_cascade.refit_on_full_validation',False)
    if args.data_csv:
        set_value(config,'data.path',str(Path(args.data_csv).resolve()))
    split_path=Path(args.split_manifest).resolve() if args.split_manifest else study/'protocol/outer_split.csv'
    if not split_path.is_file():
        raise FileNotFoundError(f'Existing outer split manifest required: {split_path}')
    set_value(config,'split.manifest',str(split_path))
    set_value(config,'split.create_if_missing',False)
    data_path=Path(config['data']['path'])
    if not data_path.is_file():
        raise FileNotFoundError(f'Private CSV missing; supply --data-csv: {data_path}')
    request=dict(stage='prepare',schema='routing-calibration-study/v1',seed=seed,variant=variant,
        input_sha256={str(p):sha(p) for p in (base/'config.resolved.yaml',base/'predictions.csv',base/'run_manifest.json',last/'predictions.csv',last/'run_manifest.json',split_path)},
        data_sha256=sha(data_path),fit_fraction=args.fit_fraction,selection_end=args.selection_end,
        source_sha256=source_inventory(project),new_base_training=False,
        calibration_separate_from_policy_selection=True)
    destination=Path(args.output_root)/'prepared'/variant/f'seed_{seed}'

    def build(out):
        print(f'[FIT small gates, CPU] {variant} seed {seed}',flush=True)
        prepared=prepare_cascade(config,str(base/'predictions.csv'),str(last/'predictions.csv'))
        frame=prepared.features.frame
        role,protocol=three_way_roles(frame,args.fit_fraction,args.selection_end)
        fit=role.eq('gate_fit').to_numpy()
        select=role.eq('policy_select').to_numpy()
        selected=select_cascade(prepared,config,fit,select,cascade_model_config(config,seed=seed))
        predictions,metrics,reference,routing,importance,route_sha=apply_selected_cascade(
            prepared,selected,model_name='followup_three_way_cascade',seed=seed)
        assert predictions.sample_id.astype(str).tolist()==frame.sample_id.astype(str).tolist()
        assert predictions.resource.astype(str).tolist()==frame.resource.astype(str).tolist()
        predictions['role']=role.to_numpy()
        hash_name=next((c for c in ('script_sha256_history','script_sha256') if c in frame),None)
        if hash_name is None or frame[hash_name].isna().any():
            raise ValueError('Exact script hash missing from causal feature context')
        predictions['script_sha256']=frame[hash_name].astype(str).to_numpy()
        predictions['script_observed']=frame['script_observed'].astype(bool).to_numpy()
        predictions.to_csv(out/'predictions.csv',index=False)
        selected.sweep.to_csv(out/'selection_sweep.csv',index=False)
        importance.to_csv(out/'feature_importance.csv',index=False)
        dump(out/'selected_policy.json',selected.policy)
        dump(out/'protocol.json',dict(**protocol,data_sha256=prepared.bundle.data_sha256,
            split_fingerprint=prepared.bundle.protocol['split_fingerprint'],
            new_base_training=False,base_checkpoint_selection='independent training tail',
            qwen_backbone=prepared.qwen_backbone,calibration_predictions_frozen=True,
            gate_fit_includes_calibration=False,selection_includes_calibration=False,
            test_used_for_fit_selection_or_calibration=False,
            note='Follow-up three-way validation protocol (default 70/15/15). Do not substitute its results silently for the main paper 70/30 protocol.'))
        dump(out/'resolved_followup_config.json',resolved_serializable_config(config))
        # Models are optional for analysis but preserve the exact lightweight fitted state.
        import joblib
        joblib.dump(selected.models,out/'cascade_models.joblib')
    publish_stage(destination,request,build,args.resume)
