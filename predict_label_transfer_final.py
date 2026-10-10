"""Complete resident-input predictors for the prospectively frozen evaluation.

No file access, gold field, fitting, parameter selection or resource download is
performed here. Every call rebuilds its necessary views and reference context;
this is also the scope of the recorded end-to-end in-memory timings.
"""
from dataclasses import dataclass
import numpy as np
from label_transfer import RidgeRanker, rank_scores, _transfer
from label_transfer_r4 import DIRECT, similarity, recurrence
from label_transfer_final import (AUX, prepare, context_for, dimensions_needed,
                                  build_features, rank_batch)


@dataclass(frozen=True)
class System:
    name: str
    kind: str
    auxiliary: str | None = None
    k: int | None = None
    dimension: int | None = None


def restore_model(record):
    return RidgeRanker(tuple(record['feature_indices']), np.array(record['mean']),
        np.array(record['scale']), np.array(record['coefficient']),
        record['intercept'], record['regularization'], record['fit_queries'],
        record['condition_number'])


def system_inventory(spec, models):
    systems=[]
    for name in models:
        aux=None
        if name.startswith('ridge_B1_all_'): aux='all'
        elif name.startswith('ridge_B1_coarse_'): aux='coarse'
        elif name.startswith('ridge_control_'): aux=name.removeprefix('ridge_control_')
        systems.append(System(name, 'ridge', aux))
    systems += [System('prior_global','prior_global'),System('prior_type','prior_type'),
                System('nearest_raw','nearest_raw'),System('nearest_centered','nearest_centered')]
    systems += [System('ordered_'+str(k),'ordered',k=k) for k in (5,10,25)]
    systems += [System(name,'vote',k=i+6) for i,name in enumerate(DIRECT)]
    systems += [System('prefix_'+str(d),'prefix',dimension=d) for d in range(2,spec['dimension']+1)]
    systems += [System('recurrence_standalone','recurrence','all'),
                System('nearest_residual','nearest_residual','residual'),
                System('vote_matched_count','matched_count','matched_count')]
    names=[s.name for s in systems]
    if len(set(names)) != len(names): raise ValueError('Duplicate system name')
    aliases={**spec['final_selections'],
             'selected_prefix':'prefix_'+str(spec['single_prefix']['selected']['dimension'])}
    if not set(aliases.values()) <= set(names): raise ValueError('Unknown selected alias')
    return systems,aliases


def _priors(records, rows, context):
    global_prior=context.counts/len(context.reference_indices)
    types=np.array([records[i].category for i in context.reference_indices])
    typed={c:(context.incidence[types==c].mean(axis=0) if (types==c).any()
              else global_prior) for c in ('Concept','Entity')}
    return global_prior, np.array([typed[records[i].category] for i in rows])


def _allowed(records, rows, context):
    result=np.ones((len(rows),len(context.candidates)),dtype=bool)
    for j,i in enumerate(rows):
        column=context.lookup.get(records[i].word)
        if column is not None:result[j,column]=False
    return result


def predict(system, matrix, available, mean, components, records, reference_rows,
            query_rows, vocabulary, models, spec, *, batch_size=128):
    """Return padded candidate IDs, coverage diagnostics, and candidate strings.

    Candidate IDs always index the independently reconstructed lexical-sorted
    dictionary. A trailing -1 is padding, never a candidate or an answer.
    """
    rows=tuple(query_rows); refs=tuple(reference_rows)
    if batch_size<1 or not rows or len(set(rows))!=len(rows):raise ValueError('Invalid query batches')
    context=context_for(records,refs,vocabulary)
    if any(records[i].word in context.reference_words for i in rows):
        raise ValueError('Query/reference lexical overlap')
    aux=() if system.auxiliary is None else (system.auxiliary,)
    dimensions=dimensions_needed(spec['dimension'],aux,
        spec['single_prefix']['selected']['dimension'],spec['variance_schedule']['dimensions'])
    if system.kind=='prefix':dimensions=(system.dimension,)
    prepared=prepare(matrix,available,mean,components,dimensions,
        spec['component_removal']['removed'] if 'residual' in aux else None)
    views=prepared.views
    usable=views.centered_valid[list(rows)] & views.centered_valid[context.reference_indices].any()
    out=np.full((len(rows),15),-1,dtype=np.int32)
    diagnostics={'usable':[], 'candidate_counts':[], 'prefix_failures':{},
                 'prefix_exact_ties':{}, 'visited_neighbor_counts':[],
                 'released_query_row':available[list(rows)].tolist(),
                 'usable_reference_rows':int(views.centered_valid[context.reference_indices].sum())}
    for start in range(0,len(rows),batch_size):
        batchrows=rows[start:start+batch_size]
        allowed=_allowed(records,batchrows,context)
        eligible=usable[start:start+len(batchrows)] & allowed.any(axis=1)
        gp,tp=_priors(records,batchrows,context)
        diagnostics['usable'].extend(eligible.tolist())
        diagnostics['candidate_counts'].extend(allowed.sum(axis=1).tolist())
        if system.kind=='ridge':
            batch=build_features(prepared,records,batchrows,context,requested=aux,
                single=spec['single_prefix']['selected']['dimension'],
                variance=spec['variance_schedule']['dimensions'])
            if not np.array_equal(batch.eligible,eligible):raise ValueError('Eligibility differs')
            prediction=rank_batch(batch,context,model=models[system.name])
            for name,neighbors in batch.neighbors.items():
                diagnostics['prefix_failures'][name]=diagnostics['prefix_failures'].get(name,0)+int((neighbors<0).sum())
                diagnostics['prefix_exact_ties'][name]=diagnostics['prefix_exact_ties'].get(name,0)+int((batch.ties[name]>1).sum())
            if 'all' in batch.neighbors:
                diagnostics['visited_neighbor_counts'].extend([len(set(v)-{-1}) for v in batch.neighbors['all']])
        elif system.kind in ('recurrence','matched_count'):
            # Direct sweep controls do not construct the unrelated B0 support,
            # typed-support, or fixed-K vote features.
            qidx=np.asarray(batchrows,dtype=np.int64)
            freq,neighbors,ties=recurrence(views,qidx,context,allowed,dimensions)
            cs=similarity(views.centered_unit,views.centered_valid,qidx,context.reference_indices)
            visited=[len(set(v)-{-1}) for v in neighbors]
            diagnostics['visited_neighbor_counts'].extend(visited)
            diagnostics['prefix_failures']['all']=diagnostics['prefix_failures'].get('all',0)+int((neighbors<0).sum())
            diagnostics['prefix_exact_ties']['all']=diagnostics['prefix_exact_ties'].get('all',0)+int((ties>1).sum())
            if system.kind=='recurrence':
                support=np.full(allowed.shape,-1.)
                validref=views.centered_valid[context.reference_indices]
                for col in range(len(context.candidates)):
                    refs_here=np.flatnonzero(context.incidence[:,col]&validref)
                    if len(refs_here):
                        maxima=cs[:,refs_here].max(axis=1)
                        support[:,col]=np.where(np.isfinite(maxima),maxima,-1.)
            else:
                order=np.argsort(-cs,axis=1,kind='stable')
            prediction=[]
            for j in range(len(batchrows)):
                if not eligible[j]:
                    prediction.append(rank_scores(tp[j],allowed[j],context.counts,secondary=tp[j]));continue
                if system.kind=='recurrence':
                    pool=np.flatnonzero(allowed[j])
                    rank=np.lexsort((pool,-context.counts[pool],-tp[j,pool],-support[j,pool],-freq[j,pool]))
                    prediction.append(tuple(pool[rank[:15]]))
                else:
                    values=context.incidence[order[j,:visited[j]]].mean(axis=0) if visited[j] else np.zeros(len(context.candidates))
                    prediction.append(rank_scores(values,allowed[j],context.counts,secondary=tp[j]))
        else:
            scores=order=None
            if system.kind not in ('prior_global','prior_type'):
                unit,valid=views.centered_unit,views.centered_valid
                if system.kind=='nearest_raw':unit,valid=views.raw_unit,views.raw_valid
                elif system.kind=='prefix':_,unit,valid=views.prefixes[0]
                elif system.kind=='nearest_residual':unit,valid=prepared.residual_unit,prepared.residual_valid
                scores=similarity(unit,valid,np.asarray(batchrows,dtype=np.int64),context.reference_indices)
                order=np.argsort(-scores,axis=1,kind='stable')
                if system.kind in ('prefix','nearest_residual'):
                    maxima=scores[np.arange(len(batchrows)),order[:,0]]
                    exists=np.isfinite(maxima)
                    diagnostics['prefix_failures'][system.name]=diagnostics['prefix_failures'].get(system.name,0)+int((~exists).sum())
                    diagnostics['prefix_exact_ties'][system.name]=diagnostics['prefix_exact_ties'].get(system.name,0)+int(((scores[exists]==maxima[exists,None]).sum(axis=1)>1).sum())
            prediction=[]
            for j in range(len(batchrows)):
                fallback=rank_scores(tp[j],allowed[j],context.counts,secondary=tp[j])
                if not eligible[j]:prediction.append(fallback);continue
                if system.kind.startswith('prior_'):
                    values=gp if system.kind=='prior_global' else tp[j]
                    prediction.append(rank_scores(values,allowed[j],context.counts,secondary=tp[j]));continue
                pool=order[j,np.isfinite(scores[j,order[j]])]
                if system.kind=='vote':
                    offset=system.k-6;k=(5,10,25)[offset%3]
                    if offset>=3:pool=np.array([x for x in pool if records[context.reference_indices[x]].category==records[batchrows[j]].category],dtype=int)
                    values=context.incidence[pool[:k]].mean(axis=0) if len(pool) else np.zeros(len(context.candidates))
                    prediction.append(rank_scores(values,allowed[j],context.counts,secondary=tp[j]))
                else:
                    k=system.k if system.kind=='ordered' else 1
                    prediction.append(_transfer(context,pool,allowed[j],k=k))
        for j,values in enumerate(prediction):
            if len(values)>15 or len(set(values))!=len(values) or any(not allowed[j,x] for x in values):
                raise ValueError('Invalid ranked output')
            out[start+j,:len(values)]=values
    return out,diagnostics,context.candidates
