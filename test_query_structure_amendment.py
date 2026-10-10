import unittest
import numpy as np
from label_transfer_final import paired_stratified_bootstrap
from query_structure_amendment import paired_signature_bootstrap,membership_groups

class MembershipBootstrapTests(unittest.TestCase):
    def test_disjoint_identity_to_frozen_algorithm(self):
        w=['a','b','a','c','d'];c=['Concept','Concept','Concept','Entity','Entity']
        y=np.arange(15,dtype=float).reshape(5,3)/17
        for batch in (1,500):
            np.testing.assert_array_equal(paired_signature_bootstrap(y,w,c,replicates=521,batch_size=batch),
                paired_stratified_bootstrap(y,w,c,replicates=521,batch_size=batch))
    def test_mixed_group_remains_paired_and_conditioned(self):
        w=['b','a','mixed','z','y','mixed'];c=['Concept','Concept','Entity','Entity','Entity','Concept']
        y=np.array([[1,-1],[2,-2],[100,-100],[4,-4],[8,-8],[200,-200]],float)
        rng=np.random.Generator(np.random.PCG64(91));expected=[]
        for start in range(0,521,500):
            count=min(500,521-start)
            cg=rng.integers(2,size=(count,2));eg=rng.integers(2,size=(count,2));mg=rng.integers(1,size=(count,1))
            for a,b,m in zip(cg,eg,mg):
                rows=[1 if k==0 else 0 for k in a]+[4 if k==0 else 3 for k in b]+[2,5]
                expected.append(y[rows].sum(axis=0)/6)
        actual=paired_signature_bootstrap(y,w,c,replicates=521,seed=91)
        np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-14)
        np.testing.assert_array_equal(actual[:,0],-actual[:,1])
        self.assertEqual(membership_groups(w,c)[-1],(('Concept','Entity'),[[2,5]]))
        # Within a one-category panel, its mixed-word occurrence is an ordinary
        # group of that panel, not held fixed by the full-population signature.
        self.assertEqual(membership_groups(['a','mixed'],['Concept','Concept']),[(('Concept',),[[0],[1]])])
    def test_mixed_only_has_no_between_lexeme_variance(self):
        a=paired_signature_bootstrap([1,5],['x','x'],['Concept','Entity'],replicates=7)
        np.testing.assert_array_equal(a,np.full((7,1),3.))

if __name__=='__main__':unittest.main()
