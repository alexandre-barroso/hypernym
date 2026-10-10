"""Meaningful synthetic prediction/metric/access checks before the test gate."""
import unittest
from unittest.mock import patch
import numpy as np
from label_transfer import Query,RidgeMoments,rank_scores
from label_transfer_final import QueryInput,prepare,context_for,build_features,AUX,rank_batch
from label_transfer_r4 import ARMS,LAMBDAS,RIDGE_NAMES,DIRECT
from predict_label_transfer_final import predict,system_inventory
from score_label_transfer_final import (score_outputs,ceiling_records,final_statistics,
                                       exact_contrast,METRICS)

class PredictionScoringTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(15009);self.nref=8;self.nq=4;d=6
        self.records=[Query('r'+str(i),'r'+str(i),'Concept' if i%2 else 'Entity',
            tuple(('common','h'+str(i),'q0')) if i%2 else ('common','h'+str(i))) for i in range(self.nref)]
        self.records += [QueryInput('q'+str(i),'q'+str(i),'Concept' if i%2 else 'Entity') for i in range(self.nq)]
        self.rows=range(self.nref,len(self.records));self.refs=range(self.nref)
        self.x=rng.normal(size=(len(self.records),d));self.av=np.ones(len(self.records),dtype=bool);self.av[-1]=False
        self.mean=rng.normal(size=d);self.u=np.linalg.qr(rng.normal(size=(d,d)))[0]
        self.vocab={'common','q0'}|{'h'+str(i) for i in self.refs}
        self.context=context_for(self.records,self.refs,self.vocab)
        self.spec={'dimension':d,'single_prefix':{'selected':{'dimension':3}},
                   'component_removal':{'removed':1},'variance_schedule':{'dimensions':[2,4,6]},
                   'final_selections':{'B0_family':'ridge_B0_g0','B0_ridge':'ridge_B0_g0',
                     'B1_all_family':'ridge_B1_all_g1','B1_all_ridge':'ridge_B1_all_g1',
                     'B1_coarse_family':'ridge_B1_coarse_g2','B1_coarse_ridge':'ridge_B1_coarse_g2'}}
        p=prepare(self.x,self.av,self.mean,self.u,range(2,d+1),1)
        self.batch=build_features(p,self.records,self.rows,self.context,requested=tuple(AUX),single=3,variance=(2,4,6))
        m=RidgeMoments(18);m.add(self.batch.features,rng.integers(2,size=self.batch.allowed.shape),self.batch.allowed,self.batch.eligible)
        self.models={name:m.fit(feature_indices=ARMS[arm],regularization=lam)
              for arm in ARMS for name,lam in zip(RIDGE_NAMES[arm],LAMBDAS)}
        for aux in ('single','residual','variance','matched_count'):
            self.models['ridge_control_'+aux]=m.fit(feature_indices=tuple(range(12))+(AUX[aux],),regularization=1.)
        self.systems,self.aliases=system_inventory(self.spec,self.models)
    def run_one(self,s,batch_size=128):
        return predict(s,self.x,self.av,self.mean,self.u,self.records,self.refs,self.rows,self.vocab,self.models,self.spec,batch_size=batch_size)
    def test_every_fitted_and_vote_output_matches_frozen_feature_ranker(self):
        for s in self.systems:
            actual,diag,c=self.run_one(s,batch_size=2)
            self.assertEqual(c,self.context.candidates)
            if s.name in self.models:expected=rank_batch(self.batch,self.context,model=self.models[s.name])
            elif s.name in DIRECT:expected=rank_batch(self.batch,self.context,feature=6+DIRECT.index(s.name))
            elif s.name=='vote_matched_count':expected=rank_batch(self.batch,self.context,feature=17)
            else:continue
            self.assertEqual([tuple(x[x>=0]) for x in actual],expected)
            self.assertEqual(diag['usable'],self.batch.eligible.tolist())
    def test_all_systems_share_fallback_self_exclusion_and_batch_invariance(self):
        fallback=None
        for s in self.systems:
            a,d,_=self.run_one(s,2);b,e,_=self.run_one(s,128)
            np.testing.assert_array_equal(a,b);self.assertEqual(d,e)
            if fallback is None:fallback=a[-1]
            np.testing.assert_array_equal(a[-1],fallback)
            self.assertNotIn(self.context.lookup['q0'],a[0])
            self.assertTrue(all(len(set(row[row>=0]))==sum(row>=0) for row in a))
    def test_direct_neighbors_and_shortlists(self):
        for kind,coords in [('nearest_raw',self.x),('nearest_centered',(self.x-self.mean)@self.u),
                            ('prefix_3',((self.x-self.mean)@self.u)[:,:3]),
                            ('nearest_residual',((self.x-self.mean)@self.u)[:,1:])]:
            s=next(s for s in self.systems if s.name==kind);a,_,_=self.run_one(s)
            for j,q in enumerate(list(self.rows)[:-1]):
                order=sorted(range(self.nref),key=lambda local:(-float(np.dot(coords[q],coords[self.context.reference_indices[local]])/
                     (np.linalg.norm(coords[q])*np.linalg.norm(coords[self.context.reference_indices[local]]))),local))
                expected=[i for i in self.context.ordered_labels[order[0]] if self.context.candidates[i]!=self.records[q].word][:15]
                self.assertEqual(a[j,a[j]>=0].tolist(),expected)
                self.assertLess(len(expected),15)
    def test_direct_sweep_controls_skip_unneeded_base_features(self):
        for name in ('recurrence_standalone','vote_matched_count'):
            s=next(s for s in self.systems if s.name==name)
            with patch('predict_label_transfer_final.build_features',side_effect=AssertionError('Unneeded B0 feature work')):
                a,_,_=self.run_one(s)
            if name=='recurrence_standalone':
                for j in range(3):
                    pool=np.flatnonzero(self.batch.allowed[j]);x=self.batch.features
                    ix=np.lexsort((pool,-self.context.counts[pool],-x[j,pool,1],-x[j,pool,2],-x[j,pool,12]))
                    self.assertEqual(a[j,a[j]>=0].tolist(),pool[ix[:15]].tolist())
    def test_metrics_scalar_denominators_and_ceiling_partition(self):
        candidate=['a','b','c'];gold=[('b','x'),tuple('abcdefghijklmnop')]
        p=np.full((1,2,15),-1,dtype=np.int32);p[0,0,:2]=[0,1];p[0,1,:3]=[2,1,0]
        m,r=score_outputs(p,candidate,gold)
        self.assertAlmostEqual(m[0,0,0],.25);self.assertAlmostEqual(m[0,1,0],3/15)
        self.assertEqual(m[0,0,1],1);self.assertAlmostEqual(m[0,0,2],1/15)
        self.assertAlmostEqual(m[0,1,5],3/16)
        ceilings=ceiling_records(['a','none'],gold,candidate,[2,1,7],{'a','b','c','x'})
        self.assertEqual(ceilings[0]['unreachable_reasons'],{'self_excluded':0,'outside_official_vocabulary':0,'no_training_label':1})
        self.assertEqual(ceilings[0]['candidate_count'],2)
        self.assertEqual(ceilings[1]['reachable_count'],3)
        broken=p.copy();broken[0,0,2]=1
        with self.assertRaises(ValueError):score_outputs(broken,candidate,gold)
        broken=p.copy();broken[0,0,1]=-1;broken[0,0,2]=1
        with self.assertRaises(ValueError):score_outputs(broken,candidate,gold)
    def test_full_statistics_identical_models_zero_primary_and_aliases(self):
        systems=[s.name for s in self.systems];gold=[('common','h0')]*4
        p=np.stack([self.run_one(s)[0] for s in self.systems]);m,_=score_outputs(p,self.context.candidates,gold)
        b=systems.index('ridge_B0_g0');a=systems.index('ridge_B1_all_g1')
        p[a]=p[b];m[a]=m[b]
        resource={'systems':systems,'aliases':self.aliases,'predictions':p,'scores':m,
            'candidates':self.context.candidates,'counts':self.context.counts,'usable':[True]*3+[False]}
        collected={}
        result=final_statistics({'glove':resource,'fasttext':resource},['q0','q1','q2','q3'],
          ['Entity','Concept','Entity','Concept'],gold,{'glove':self.spec,'fasttext':self.spec},self.vocab,
          replicates=30,sink=lambda n,v:collected.update({n:v}))
        for name in ('glove','fasttext'):
            primary=result['resources'][name]['contrasts']['primary']
            self.assertEqual(primary['difference']['ap_capped_gold'],0)
            self.assertEqual(primary['bonferroni_primary_97_5_interval'],[0,0])
            self.assertEqual(primary['ap_ties'],4)
            self.assertEqual(primary['identical_lists'],4)
            self.assertEqual(result['resources'][name]['panels']['fallback']['queries'],1)
        self.assertEqual(len(collected['all_query_bootstrap']['descriptors']),58)

if __name__=='__main__':unittest.main()
