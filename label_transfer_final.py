"""Original final-evaluation primitives: no file access and no query gold input.

The feature builder only reads labels in an explicitly constructed reference
context. QueryInput deliberately has no labels attribute. Numerical choices
match the training-only R4 core; additional controls have fixed definitions.
"""
from dataclasses import dataclass
import numpy as np
from label_transfer import Views, LabelContext, _unit, make_context, rank_scores, RidgeMoments
from label_transfer_r4 import FEATURES as CORE_FEATURES, similarity, recurrence, coarse_schedule

FEATURES=CORE_FEATURES+('selected_prefix_list_indicator','centered_component_removal_list_indicator',
                      'variance_spaced_recurrence','sweep_visited_count_full_space_vote')
AUX={'all':12,'coarse':13,'single':14,'residual':15,'variance':16,'matched_count':17}

@dataclass(frozen=True)
class QueryInput:
    identifier:str
    word:str
    category:str
    def __post_init__(self):
        if not self.identifier or not self.word or self.word!=self.word.lower():
            raise ValueError('Nonempty identifier and lowercase lexical string required')
        if self.category not in ('Concept','Entity'):raise ValueError('Unknown query type')

@dataclass(frozen=True)
class Prepared:
    views:Views
    residual_unit:np.ndarray|None
    residual_valid:np.ndarray|None

@dataclass
class FinalBatch:
    query_rows:np.ndarray
    features:np.ndarray
    allowed:np.ndarray
    eligible:np.ndarray
    centered_scores:np.ndarray
    centered_order:np.ndarray
    neighbors:dict
    ties:dict

def prepare(matrix,available,mean,components,dimensions=(),remove_components=None):
    values=np.asarray(matrix,dtype=np.float64);available=np.asarray(available)
    mean=np.asarray(mean,dtype=np.float64);u=np.asarray(components,dtype=np.float64)
    if values.ndim!=2 or not len(values) or available.shape!=(len(values),) or available.dtype.kind!='b':
        raise ValueError('Invalid matrix or availability')
    d=values.shape[1]
    if mean.shape!=(d,) or u.shape!=(d,d) or not all(np.isfinite(a).all() for a in (values,mean,u)):
        raise ValueError('Invalid finite PCA inputs')
    if not np.allclose(u.T@u,np.eye(d),rtol=0,atol=1e-8):raise ValueError('Nonorthonormal basis')
    dimensions=tuple(dimensions)
    if len(set(dimensions))!=len(dimensions) or any(isinstance(k,bool) or not isinstance(k,(int,np.integer)) or not 2<=k<=d for k in dimensions):
        raise ValueError('Invalid prefix schedule')
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        z=(values-mean)@u
        raw,raw_ok=_unit(values,available);full,full_ok=_unit(z,available)
        prefixes=tuple((int(k),*_unit(z[:,:k],available)) for k in dimensions)
        residual=valid=None
        if remove_components is not None:
            if isinstance(remove_components,bool) or not isinstance(remove_components,(int,np.integer)) or not 1<=remove_components<d:
                raise ValueError('Invalid removed-component count')
            residual,valid=_unit(z[:,remove_components:],available)
    return Prepared(Views(raw,raw_ok,full,full_ok,prefixes,available.copy()),residual,valid)

def context_for(records,reference_rows,vocabulary):
    rows=tuple(reference_rows)
    if not rows or len(set(rows))!=len(rows):raise ValueError('Distinct nonempty reference rows required')
    if any(label in vocabulary for i in rows for label in records[i].labels):
        return make_context(records,rows,vocabulary)
    rows=tuple(sorted(rows,key=lambda i:(records[i].word,records[i].identifier)))
    return LabelContext(np.array(rows,dtype=np.int64),frozenset(records[i].word for i in rows),(),{},
                        np.zeros(0,dtype=np.int64),np.zeros((len(rows),0),dtype=bool),tuple(() for i in rows))

def dimensions_needed(d,requested,single,variance):
    wanted=set()
    for feature in requested:
        if feature in ('all','matched_count'):wanted.update(range(2,d+1))
        elif feature=='coarse':wanted.update(coarse_schedule(d))
        elif feature=='single':wanted.add(single)
        elif feature=='variance':wanted.update(variance)
        elif feature!='residual':raise ValueError('Unknown auxiliary feature')
    return tuple(sorted(wanted))

def build_features(prepared,records,query_rows,context,*,requested=(),single=None,variance=()):
    rows=np.asarray(tuple(query_rows),dtype=np.int64);v=prepared.views
    if not len(rows) or len(set(rows.tolist()))!=len(rows):raise ValueError('Distinct query rows required')
    if any(records[i].word in context.reference_words for i in rows):raise ValueError('Query/reference lexical overlap')
    requested=tuple(requested)
    if len(set(requested))!=len(requested) or not set(requested)<=set(AUX):raise ValueError('Invalid requested feature set')
    allowed=np.ones((len(rows),len(context.candidates)),dtype=bool)
    for r,i in enumerate(rows):
        own=context.lookup.get(records[i].word)
        if own is not None:allowed[r,own]=False
    x=np.zeros((*allowed.shape,len(FEATURES)))
    scores=similarity(v.centered_unit,v.centered_valid,rows,context.reference_indices)
    order=np.argsort(-scores,axis=1,kind='stable')
    valid=v.centered_valid[context.reference_indices]
    eligible=v.centered_valid[rows]&valid.any()&allowed.any(axis=1)
    types=np.array([records[i].category for i in context.reference_indices]);qt=np.array([records[i].category for i in rows])
    x[:,:,0]=context.counts/len(context.reference_indices)
    for cat in ('Concept','Entity'):
        group=types==cat;idx=np.flatnonzero(qt==cat)
        if len(idx):x[idx,:,1]=context.incidence[group].mean(axis=0) if group.any() else x[0,:,0]
    for col in range(len(context.candidates)):
        support=np.flatnonzero(context.incidence[:,col]&valid)
        if len(support):
            maxima=scores[:,support].max(axis=1);x[:,col,2]=np.where(np.isfinite(maxima),maxima,-1.);x[:,col,3]=1.
        else:x[:,col,2]=-1.
        for cat in ('Concept','Entity'):
            idx=np.flatnonzero(qt==cat);local=support[types[support]==cat]
            if len(local):
                maxima=scores[np.ix_(idx,local)].max(axis=1);x[idx,col,4]=np.where(np.isfinite(maxima),maxima,-1.);x[idx,col,5]=1.
            else:x[idx,col,4]=-1.
    for row in np.flatnonzero(eligible):
        global_order=order[row,:int(valid.sum())];typed=global_order[types[global_order]==qt[row]]
        for offset,pool in ((6,global_order),(9,typed)):
            for j,k in enumerate((5,10,25)):
                if len(pool):x[row,:,offset+j]=context.incidence[pool[:k]].mean(axis=0)
    neighbors={};ties={};d=v.centered_unit.shape[1]
    schedules={'all':tuple(range(2,d+1)),'coarse':coarse_schedule(d),'single':(single,), 'variance':tuple(variance)}
    compute=set(requested)
    if 'matched_count' in compute:compute.add('all')
    for name in ('all','coarse','single','variance'):
        if name not in compute:continue
        x[:,:,AUX[name]],neighbors[name],ties[name]=recurrence(v,rows,context,allowed,schedules[name])
    if 'residual' in compute:
        if prepared.residual_unit is None:raise ValueError('Residual view required')
        residual=Views(v.raw_unit,v.raw_valid,v.centered_unit,v.centered_valid,
                       ((d,prepared.residual_unit,prepared.residual_valid),),v.available)
        x[:,:,15],neighbors['residual'],ties['residual']=recurrence(residual,rows,context,allowed,(d,))
    if 'matched_count' in compute:
        for row in np.flatnonzero(eligible):
            m=len(set(neighbors['all'][row])-{-1})
            if m:x[row,:,17]=context.incidence[order[row,:m]].mean(axis=0)
    return FinalBatch(rows,x,allowed,eligible,scores,order,neighbors,ties)

def rank_batch(batch,context,model=None,feature=None,depth=15):
    if (model is None)==(feature is None):raise ValueError('Exactly one rank source required')
    scores=model.predict(batch.features) if model is not None else batch.features[:,:,feature]
    out=[]
    for row in range(len(batch.query_rows)):
        s=scores[row] if batch.eligible[row] else batch.features[row,:,1]
        out.append(rank_scores(s,batch.allowed[row],context.counts,depth=depth,secondary=batch.features[row,:,1]))
    return out

def paired_stratified_bootstrap(values,words,categories,*,replicates=100000,seed=2026092605,batch_size=500):
    """Lexical-group resampling with fixed observed stratum row weights.

    All supplied contrasts share each draw. Coefficients/fits stay fixed.
    Values are per-row paired metric differences, never pseudo-independent labels.
    """
    y=np.asarray(values,dtype=np.float64)
    if y.ndim==1:y=y[:,None]
    if y.ndim!=2 or not len(y) or len(words)!=len(y) or len(categories)!=len(y) or not np.isfinite(y).all():
        raise ValueError('Invalid aligned paired values')
    if replicates<1 or batch_size<1:raise ValueError('Positive replicate and batch counts required')
    strata={};word_type={}
    for i,(word,cat) in enumerate(zip(words,categories)):
        if cat not in ('Concept','Entity'):raise ValueError('Invalid stratum')
        if word in word_type and word_type[word]!=cat:raise ValueError('Conflicting lexical strata')
        word_type[word]=cat;strata.setdefault(cat,{}).setdefault(word,[]).append(i)
    groups=[]
    for cat in ('Concept','Entity'):
        if cat not in strata:continue
        index=[strata[cat][w] for w in sorted(strata[cat])]
        sums=np.stack([y[i].sum(axis=0) for i in index]);sizes=np.array([len(i) for i in index])
        groups.append((sums,sizes,sizes.sum()/len(y)))
    rng=np.random.Generator(np.random.PCG64(seed));draws=np.zeros((replicates,y.shape[1]))
    for start in range(0,replicates,batch_size):
        n=min(batch_size,replicates-start)
        for sums,sizes,weight in groups:
            idx=rng.integers(len(sizes),size=(n,len(sizes)))
            draws[start:start+n]+=weight*sums[idx].sum(axis=1)/sizes[idx].sum(axis=1)[:,None]
    return draws
