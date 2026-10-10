"""Independent finite checks before any official held-out input is accessed."""
import unittest
import numpy as np
from label_transfer import Query, RidgeMoments, prepare_views
from label_transfer_r4 import build_features as old_features, ARMS, LAMBDAS
from label_transfer_final import (AUX, QueryInput, prepare, context_for, dimensions_needed,
    build_features, rank_batch, paired_stratified_bootstrap)

class FinalProtocolTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(801)
        self.records=[Query('r'+str(i),'ref'+str(i),'Concept' if i%2 else 'Entity',
                       tuple(['shared','label'+str(i),'query0'] if i%2 else ['shared','label'+str(i)])) for i in range(6)]
        self.records += [QueryInput('q'+str(i),'query'+str(i),'Entity' if i%2 else 'Concept') for i in range(3)]
        self.x=self.rng.normal(size=(9,6));self.mu=self.rng.normal(size=6)
        self.u=np.linalg.qr(self.rng.normal(size=(6,6)))[0]
        self.available=np.ones(9,dtype=bool);self.available[-1]=False
        self.context=context_for(self.records,range(6),{'shared','query0'}|{'label'+str(i) for i in range(6)})
        self.queries=[6,7,8];self.schedules=dict(requested=tuple(AUX),single=3,variance=(2,4,6))
        self.prepared=prepare(self.x,self.available,self.mu,self.u,range(2,7),1)

    def test_exact_core_agreement_with_frozen_r4(self):
        old=prepare_views(self.x,self.available,mean=self.mu,components=self.u,dimensions=range(2,7))
        a=old_features(old,self.records,self.queries,self.context)
        b=build_features(self.prepared,self.records,self.queries,self.context,**self.schedules)
        np.testing.assert_array_equal(a.features,b.features[:,:,:14])
        np.testing.assert_array_equal(a.allowed,b.allowed)
        np.testing.assert_array_equal(a.eligible,b.eligible)
        np.testing.assert_array_equal(a.neighbors_all,b.neighbors['all'])
        self.assertFalse(hasattr(self.records[6],'labels'))

    def test_scalar_auxiliaries_and_centered_residual(self):
        b=build_features(self.prepared,self.records,self.queries,self.context,**self.schedules)
        z=(self.x-self.mu)@self.u
        projection=(self.x-self.mu)-(self.x-self.mu)@self.u[:,:1]@self.u[:,:1].T
        np.testing.assert_allclose(np.linalg.norm(projection,axis=1),np.linalg.norm(z[:,1:],axis=1),rtol=1e-13,atol=1e-13)
        def transfer(q,coords):
            if not self.available[q] or np.linalg.norm(coords[q])==0:return (),-1
            values=[]
            for loc,ref in enumerate(self.context.reference_indices):
                if self.available[ref] and np.linalg.norm(coords[ref]):
                    sim=float(np.dot(coords[q],coords[ref])/(np.linalg.norm(coords[q])*np.linalg.norm(coords[ref])))
                    values.append((sim,loc,ref))
            best=sorted(values,key=lambda x:(-x[0],x[1]))[0]
            labels=[j for j in self.context.ordered_labels[best[1]] if self.context.candidates[j]!=self.records[q].word][:15]
            return labels,best[2]
        for row,q in enumerate(self.queries):
            expected=np.zeros((len(self.context.candidates),6));neighbors=[]
            for col,schedule in [(0,range(2,7)),(1,range(2,7)),(2,(3,)),(4,(2,4,6))]:
                for dim in schedule:
                    labels,neighbor=transfer(q,z[:,:dim]);expected[labels,col]+=1/len(schedule)
                    if col==0:neighbors.append(neighbor)
            labels,_=transfer(q,z[:,1:]);expected[labels,3]=1
            m=len(set(neighbors)-{-1})
            if m:
                # Full-space neighbor order is recomputed independently.
                sim=[np.dot(z[q],z[r])/(np.linalg.norm(z[q])*np.linalg.norm(z[r])) for r in self.context.reference_indices]
                order=sorted(range(len(sim)),key=lambda j:(-sim[j],j))[:m]
                expected[:,5]=self.context.incidence[order].sum(axis=0)/m
            np.testing.assert_allclose(b.features[row,:,12:],expected,rtol=0,atol=2e-16)

    def test_subset_views_do_not_compute_sweep(self):
        self.assertEqual(dimensions_needed(6,(),3,(2,4,6)),())
        self.assertEqual(dimensions_needed(6,('single',),3,(2,4,6)),(3,))
        p=prepare(self.x,self.available,self.mu,self.u,(),None)
        b=build_features(p,self.records,self.queries,self.context)
        self.assertEqual(b.neighbors,{})
        np.testing.assert_array_equal(b.features[:,:,12:],0)
        dense=build_features(self.prepared,self.records,self.queries,self.context,**self.schedules)
        np.testing.assert_array_equal(b.features[:,:,:12],dense.features[:,:,:12])

    def test_common_fallback_and_no_self_candidate(self):
        b=build_features(self.prepared,self.records,self.queries,self.context,**self.schedules)
        a=rank_batch(b,self.context,feature=6);c=rank_batch(b,self.context,feature=12)
        self.assertEqual(a[-1],c[-1])
        self.assertNotIn(self.context.lookup['query0'],a[0])
        self.assertNotIn(self.context.lookup['query0'],c[0])
        self.assertFalse(b.eligible[-1])

    def test_empty_candidate_inventory(self):
        c=context_for(self.records,range(6),set())
        b=build_features(self.prepared,self.records,self.queries,c,**self.schedules)
        self.assertEqual(rank_batch(b,c,feature=12),[(),(),()])
        self.assertFalse(b.eligible.any())

    def test_leakage_rejected(self):
        with self.assertRaisesRegex(ValueError,'overlap'):
            build_features(self.prepared,self.records,[0],self.context)

    def test_all_ridge_arms_match_direct_weighted_solve(self):
        x=self.rng.normal(size=(8,7,18));x[:,:,2]=1.
        y=self.rng.integers(0,2,size=(8,7));allowed=self.rng.random((8,7))>.2;eligible=np.ones(8,dtype=bool);eligible[-1]=False
        m=RidgeMoments(18);m.add(x,y,allowed,eligible)
        xx=np.concatenate([x[i,allowed[i]] for i in range(7)]);yy=np.concatenate([y[i,allowed[i]] for i in range(7)])
        weights=np.concatenate([np.full(allowed[i].sum(),1/(7*allowed[i].sum())) for i in range(7)])
        specs=[(cols,lam) for cols in ARMS.values() for lam in LAMBDAS]+[(tuple(range(12))+(j,),1.) for j in range(14,18)]
        self.assertEqual(len(specs),19)
        for columns,lam in specs:
            fitted=m.fit(feature_indices=columns,regularization=lam)
            design=xx[:,columns];mean=np.sum(weights[:,None]*design,axis=0)
            var=np.sum(weights[:,None]*(design-mean)**2,axis=0);scale=np.sqrt(var)
            raw=np.sum(weights[:,None]*design**2,axis=0);scale[var<=64*np.finfo(float).eps*np.maximum(1,raw)]=1.
            standard=(design-mean)/scale;aug=np.column_stack([np.ones(len(xx)),standard])
            ridge=np.diag([0.]+[lam]*len(columns));solution=np.linalg.solve(aug.T@(weights[:,None]*aug)+ridge,aug.T@(weights*yy))
            np.testing.assert_allclose(fitted.mean,mean,rtol=0,atol=1e-14)
            np.testing.assert_allclose(np.r_[fitted.intercept,fitted.coefficient],solution,rtol=1e-8,atol=1e-10)

    def test_bootstrap_pairing_reversal_and_fixed_strata(self):
        words=['a','a','b','c','d'];cats=['Concept']*3+['Entity']*2
        y=np.array([1.,2.,3.,4.,5.]);matrix=np.column_stack([y,y,np.zeros(5),-y])
        a=paired_stratified_bootstrap(matrix,words,cats,replicates=1000,seed=21,batch_size=20)
        np.testing.assert_array_equal(a[:,0],a[:,1]);np.testing.assert_array_equal(a[:,2],0)
        np.testing.assert_allclose(a[:,0],-a[:,3],rtol=0,atol=0)
        fixed=paired_stratified_bootstrap([1.,1.,1.,10.,10.],words,cats,replicates=100,seed=21)
        np.testing.assert_allclose(fixed,4.6,atol=1e-15)
        quant=np.quantile(a[:,0],[.0125,.9875],method='linear')
        neg=np.quantile(a[:,3],[.0125,.9875],method='linear')
        np.testing.assert_allclose(quant,-neg[::-1],atol=1e-14)

    def test_bootstrap_duplicate_group_reference_replay(self):
        words=['a','a','b','c','d'];cats=['Concept']*3+['Entity']*2;y=np.arange(5.)
        actual=paired_stratified_bootstrap(y,words,cats,replicates=9,seed=77,batch_size=9)[:,0]
        rng=np.random.Generator(np.random.PCG64(77));c=rng.integers(2,size=(9,2));e=rng.integers(2,size=(9,2));expected=[]
        for left,right in zip(c,e):
            ci=[i for group in left for i in ([0,1] if group==0 else [2])]
            ei=[3+group for group in right]
            expected.append(.6*y[ci].mean()+.4*y[ei].mean())
        np.testing.assert_allclose(actual,expected,atol=1e-15)
        with self.assertRaisesRegex(ValueError,'Conflicting'):
            paired_stratified_bootstrap([0.,1.],['a','a'],['Concept','Entity'],replicates=2)

if __name__=='__main__':unittest.main()
