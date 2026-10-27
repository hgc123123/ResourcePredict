import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import shutil
from uuid import uuid4

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from study_utils import (three_way_roles,cluster_difference_ci,publish_stage,dump,read_prepared)
from gate_study import (fit_cutoff,apply_cutoff,cache_counts,score,tie_uniform,gate_one)
from allocation_study import (conformal_margin,apply_margin,fit_bound,joint_metrics,allocation_one)


@contextmanager
def working_temp():
    # Some Windows file permissions make tempfile's mode-0700 directory unreadable.
    parent=ROOT/'qa_tmp'
    parent.mkdir(exist_ok=True)
    p=parent/uuid4().hex
    p.mkdir()
    try:
        yield str(p)
    finally:
        assert p.resolve().parent==parent.resolve() and len(p.name)==32
        shutil.rmtree(p)


def fixture():
    rng=np.random.default_rng(42)
    rows=[]
    for j in range(260):
        submit=pd.Timestamp('2026-01-01',tz='UTC')+pd.Timedelta(hours=j*3)
        y=float(rng.uniform(.05,.95))
        for resource in ('memory','vram'):
            history=j%4!=0
            exact=j%4 in (1,2)
            q=float(np.clip(y+rng.normal(0,.08),0,1))
            post=float(np.clip(y+rng.normal(0,.04),0,1))
            h=float(np.clip(y+rng.normal(0,.03 if exact else .12),0,1))
            rows.append(dict(sample_id=f'j{j:04d}',resource=resource,user_id=f'u{j%5}',
                split='validation' if j<200 else 'test',submit_time=submit,
                completion_time=submit+pd.Timedelta(minutes=30),target=y,
                prediction=h if exact else post,qwen_prediction=q,post_blend_prediction=post,
                hierarchical_reference_prediction=h if history else q,history_expert_prediction=h if history else np.nan,
                reliability_probability=float(rng.uniform()),history_available=history,
                exact_script_history_available=exact,resource_applicable=True,valid=True,
                exact_history_count=j%6+1,exact_history_age_hours=j%9,
                exact_history_mad5=(j%4)*.01,script_sha256=f'key{j%8}',script_observed=True))
    d=pd.DataFrame(rows)
    role,_=three_way_roles(d)
    d['role']=role
    return d


class ProtocolTests(unittest.TestCase):
    def test_disjoint_calibration_and_completion_purge(self):
        f=fixture()
        boundary=f.loc[f.role.eq('policy_select'),'submit_time'].min()
        first=f.index[f.role.eq('gate_fit')][0]
        identity=f.loc[first,'sample_id']
        f.loc[f.sample_id.eq(identity),'completion_time']=boundary
        role,meta=three_way_roles(f)
        self.assertTrue(role[f.sample_id.eq(identity)].eq('gate_fit_purged').all())
        self.assertEqual(meta['unique_jobs']['calibration'],30)
        self.assertFalse((role.eq('calibration')&role.eq('policy_select')).any())

    def test_target_changes_do_not_change_roles(self):
        f=fixture();r,_=three_way_roles(f)
        f['target']=1-f.target
        s,_=three_way_roles(f)
        self.assertTrue(r.equals(s))

    def test_target_free_scores_and_cutoffs(self):
        f=fixture();g=f.copy();g['target']=1-f.target;g['qwen_prediction']=0
        for name in ('learned','recent_history','low_dispersion','many_observations','random_hash'):
            np.testing.assert_equal(score(f,name),score(g,name))
        s=np.array([.5,.5,.5,.5]);u=np.array([.1,.2,.3,.4])
        cut=fit_cutoff(s,u,.5)
        np.testing.assert_equal(apply_cutoff(s,u,cut),[False,False,True,True])
        self.assertEqual(cut['selection_bypass'],2)

    def test_no_test_quantile_used(self):
        cut=fit_cutoff(np.array([.1,.2,.8,.9]),np.arange(4)/4,.5)
        a=apply_cutoff(np.array([.7,.95]),np.array([.1,.1]),cut)
        b=apply_cutoff(np.array([.7,.95,.999]),np.array([.1,.1,.1]),cut)
        np.testing.assert_equal(a,b[:2])

    def test_cache_same_timestamp_delayed_insertion(self):
        f=fixture().iloc[:4].copy()
        f['split']='test';f['role']='test';f['submit_time']=f.submit_time.min();f['script_sha256']='same'
        c=cache_counts(f,np.zeros(len(f),bool),1)
        self.assertEqual(int(c.encoder_forward.sum()),2)
        self.assertEqual(int(c.cache_hit.sum()),0)

    def test_cache_not_primed_with_training_rows(self):
        f=fixture().iloc[:4].copy()
        f['script_sha256']='same';f.loc[f.sample_id.eq('j0000'),'split']='train'
        f.loc[f.sample_id.eq('j0000'),'role']='excluded'
        f.loc[f.sample_id.eq('j0001'),['split','role']]='test'
        c=cache_counts(f,np.zeros(len(f),bool),1)
        self.assertEqual(int(c.encoder_forward.sum()),1)

    def test_selection_rows_do_not_prime_postfreeze_cache(self):
        f=fixture().iloc[:4].copy()
        f['script_sha256']='same'
        f.loc[f.sample_id.eq('j0000'),['split','role']]=['validation','policy_select']
        f.loc[f.sample_id.eq('j0001'),['split','role']]='test'
        c=cache_counts(f,np.zeros(len(f),bool),1)
        self.assertEqual(int(c.encoder_forward.sum()),1)

    def test_zero_paired_difference(self):
        x=np.linspace(.1,.9,20)
        result=cluster_difference_ci(x,x+.01,x+.01,np.repeat(np.arange(4),5),replicates=20)
        for r in result:
            self.assertEqual(r['delta'],0)
            self.assertEqual(r['ci_lower'],0)
            self.assertEqual(r['ci_upper'],0)

    def test_conformal_finite_sample_rank_and_infinity(self):
        finite=conformal_margin(np.arange(1,20)/100,.05)
        self.assertEqual(finite['rank'],19)
        self.assertEqual(finite['margin'],.19)
        inf=conformal_margin(np.arange(1,20)/100,.01)
        self.assertTrue(inf['full_bound'])
        np.testing.assert_equal(apply_margin([.2,.8],inf),[1.,1.])

    def test_sparse_mondrian_groups_fail_closed(self):
        f=fixture();cal=f.role.eq('calibration').to_numpy()
        upper,records=fit_bound(f,f.prediction.to_numpy(),cal,.025,'mondrian_history',30)
        cold=~f.history_available
        self.assertTrue(np.all(upper[cold]==1))
        self.assertTrue(any(r['reason']=='sparse_history_group_full_bound' for r in records))

    def test_calibration_does_not_read_test_targets(self):
        f=fixture();g=f.copy();g.loc[g.role.eq('test'),'target']=1-g.loc[g.role.eq('test'),'target']
        cal=f.role.eq('calibration').to_numpy()
        for kind in ('global','mondrian_history','day_max'):
            u,a=fit_bound(f,f.prediction.to_numpy(),cal,.025,kind)
            v,b=fit_bound(g,g.prediction.to_numpy(),cal,.025,kind)
            np.testing.assert_equal(u,v);self.assertEqual(a,b)

    def test_joint_unknown_labels_are_not_counted_as_successes(self):
        f=fixture().loc[lambda x:x.role.eq('test')].copy().reset_index(drop=True)
        f.loc[0,'valid']=False
        r=joint_metrics(f,np.ones(len(f)),np.ones(len(f),bool),42,'x','unit',.05,20,2026)
        self.assertEqual(r['excluded_unknown_jobs'],1)
        self.assertEqual(r['complete_label_jobs'],59)
        self.assertEqual(r['joint_shortfall_rate'],0)
        self.assertTrue(r['degenerate_bootstrap'])

    def test_all_joint_labels_missing_returns_unknown(self):
        f=fixture().loc[lambda x:x.role.eq('test')].copy().reset_index(drop=True)
        f['valid']=False
        r=joint_metrics(f,np.ones(len(f)),np.ones(len(f),bool),42,'x','unit',.05,20,2026)
        self.assertIsNone(r['joint_shortfall_rate'])
        self.assertIsNone(r['empirical_joint_target_met'])

    def test_resume_rejects_changed_request_and_corruption(self):
        with working_temp() as tmp:
            dest=Path(tmp)/'run'
            publish_stage(dest,{'a':1},lambda p:dump(p/'value.json',{'v':3}))
            publish_stage(dest,{'a':1},lambda p:self.fail('Should skip'),resume=True)
            with self.assertRaises(ValueError):
                publish_stage(dest,{'a':2},lambda p:None,resume=True)
            (dest/'value.json').write_text('changed')
            with self.assertRaises(ValueError):
                publish_stage(dest,{'a':1},lambda p:None,resume=True)

    def test_synthetic_analysis_end_to_end(self):
        with working_temp() as tmp:
            output=Path(tmp)
            f=fixture()
            for variant in ('full','no_code'):
                def writer(p,variant=variant):
                    data=f.copy()
                    if variant=='no_code':
                        data['prediction']=np.clip(data.prediction+.01,0,1)
                    data.to_csv(p/'predictions.csv',index=False)
                publish_stage(output/'prepared'/variant/'seed_42',{'synthetic':True,'variant':variant},writer)
            args=argparse.Namespace(output_root=output,variants=['full','no_code'],seeds=[42],
                budgets=[0.,.5,1.],bootstrap_replicates=10,bootstrap_seed=2026,cache_capacity=4,
                alphas=[.05],min_group_n=5,resume=False)
            gate_one(args,42)
            allocation_one(args,42)
            self.assertTrue((output/'gate/full/seed_42/gate_paired_intervals.csv').is_file())
            r=pd.read_csv(output/'allocation/seed_42/joint_bounds_metrics.csv')
            self.assertTrue(r.loc[r.calibrator.eq('unit_bound'),'joint_shortfall_rate'].eq(0).all())


if __name__=='__main__':
    unittest.main()
