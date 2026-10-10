"""Query-discovered scope and grouping amendment, before predictions/gold.

Training, model choice and prediction rules are unchanged. This module extends
only paired uncertainty to lexical groups whose observed category sets differ.
"""
import numpy as np

SIGNATURE_ORDER=(('Concept',),('Entity',),('Concept','Entity'))

def membership_groups(words,categories):
    if len(words)!=len(categories):raise ValueError('Unaligned lexical identities')
    by_word={}
    for i,(word,cat) in enumerate(zip(words,categories)):
        if cat not in ('Concept','Entity'):raise ValueError('Invalid stratum')
        by_word.setdefault(word,[]).append(i)
    signatures={}
    for word,rows in by_word.items():
        signature=tuple(sorted({categories[i] for i in rows}))
        signatures.setdefault(signature,{})[word]=rows
    return [(signature,[signatures[signature][w] for w in sorted(signatures[signature])])
            for signature in SIGNATURE_ORDER if signature in signatures]

def paired_signature_bootstrap(values,words,categories,*,replicates=100000,seed=2026092605,batch_size=500):
    """Fixed membership-signature row weights; whole lexical groups stay paired.

    The observed mixed-category singleton is conditioned on, not a source of
    estimable between-lexeme variance. Recompute signatures within each panel.
    With no mixed group, RNG draws and values match the original procedure.
    """
    y=np.asarray(values,dtype=np.float64)
    if y.ndim==1:y=y[:,None]
    if y.ndim!=2 or not len(y) or len(words)!=len(y) or len(categories)!=len(y) or not np.isfinite(y).all():
        raise ValueError('Invalid aligned paired values')
    if replicates<1 or batch_size<1:raise ValueError('Positive replicate and batch counts required')
    groups=[]
    for signature,index in membership_groups(words,categories):
        sums=np.stack([y[i].sum(axis=0) for i in index]);sizes=np.array([len(i) for i in index])
        groups.append((sums,sizes,sizes.sum()/len(y)))
    rng=np.random.Generator(np.random.PCG64(seed));draws=np.zeros((replicates,y.shape[1]))
    for start in range(0,replicates,batch_size):
        n=min(batch_size,replicates-start)
        for sums,sizes,weight in groups:
            idx=rng.integers(len(sizes),size=(n,len(sizes)))
            draws[start:start+n]+=weight*sums[idx].sum(axis=1)/sizes[idx].sum(axis=1)[:,None]
    return draws
