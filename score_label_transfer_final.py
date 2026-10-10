"""Gold-dependent final scoring, deliberately separate from prediction code."""
import numpy as np
from retrieval_metrics import evaluate
from query_structure_amendment import paired_signature_bootstrap as paired_stratified_bootstrap, membership_groups

METRICS=('ap_capped_gold','hits','precision','recall','reciprocal_rank','ap_full_gold',
         'semeval_mean_normalized_prefix_precision','semeval_normalized_precision','output_length')
PANELS=('usable','fallback','Concept','Entity','usable Concept','usable Entity',
        'common','common Concept','common Entity')


def score_outputs(predictions, candidates, gold):
    a=np.asarray(predictions)
    if a.ndim!=3 or a.shape[1]!=len(gold) or a.shape[2]!=15 or a.dtype.kind not in 'iu':
        raise ValueError('Expected systems by aligned queries by depth-15 IDs')
    if len(set(candidates))!=len(candidates):raise ValueError('Duplicate candidate')
    result=np.zeros((a.shape[0],a.shape[1],len(METRICS)))
    relevance=np.zeros(a.shape,dtype=np.int8)
    for s in range(len(a)):
        for q in range(len(gold)):
            ids=a[s,q];valid=ids[ids>=0]
            if (ids < -1).any() or (valid>=len(candidates)).any() or len(set(valid))!=len(valid):
                raise ValueError('Invalid candidate IDs')
            if not np.array_equal(ids[:len(valid)],valid):raise ValueError('Nontrailing output padding')
            m=evaluate([candidates[x] for x in valid],gold[q],depth=15)
            relevance[s,q]=m.relevance
            result[s,q]=[getattr(m,k) if k not in ('hits','output_length')
                         else (sum(m.relevance) if k=='hits' else len(m.predictions)) for k in METRICS]
    return result,relevance


def ceiling_records(words, gold, candidates, counts, vocabulary):
    known={word:int(count) for word,count in zip(candidates,counts)}
    result=[]
    for word,labels in zip(words,gold):
        if not labels or len(set(labels))!=len(labels):raise ValueError('Empty/duplicate gold')
        reachable=[x for x in labels if x in known and x!=word]
        reasons={'self_excluded':0,'outside_official_vocabulary':0,'no_training_label':0}
        for x in labels:
            if x==word:reasons['self_excluded']+=1
            elif x not in vocabulary:reasons['outside_official_vocabulary']+=1
            elif x not in known:reasons['no_training_label']+=1
        if len(reachable)+sum(reasons.values())!=len(labels):raise ValueError('Ceiling partition error')
        hit=min(15,len(reachable));g=len(labels)
        result.append({'gold_count':g,'reachable_count':len(reachable),
            'candidate_count':len(candidates)-int(word in known),'unreachable_reasons':reasons,
            'ap_capped_gold':hit/min(15,g),'precision':hit/15,'recall':hit/g,
            'frequency_bands':{band:sum(frequency_band(known.get(x,0))==band for x in labels)
                               for band in ('0','1','2-5','6+')}})
    return result


def frequency_band(n):return '0' if n==0 else '1' if n==1 else '2-5' if n<=5 else '6+'


def masks(categories, usable, common):
    cats=np.asarray(categories);u=np.asarray(usable,dtype=bool);c=np.asarray(common,dtype=bool)
    return {'all':np.ones(len(cats),dtype=bool),'usable':u,'fallback':~u,
        'Concept':cats=='Concept','Entity':cats=='Entity',
        'usable Concept':u&(cats=='Concept'),'usable Entity':u&(cats=='Entity'),
        'common':c,'common Concept':c&(cats=='Concept'),'common Entity':c&(cats=='Entity')}


def contrast_specs(resource, spec):
    selections=spec['final_selections'];base=selections['B0_family'];all_=selections['B1_all_family'];coarse=selections['B1_coarse_family']
    contrast=[{'name':'primary','left':all_,'right':base,'primary':True},
              {'name':'coarse_family_minus_base','left':coarse,'right':base,'primary':False},
              {'name':'all_family_minus_coarse','left':all_,'right':coarse,'primary':False}]
    for j in range(5):
        for arm in ('all','coarse'):
            contrast.append({'name':f'{arm}_minus_base_penalty_{j}',
                'left':f'ridge_B1_{arm}_g{j}','right':f'ridge_B0_g{j}','primary':False})
    j=int(base.rsplit('g',1)[1]);at_base=f'ridge_B1_all_g{j}'
    for aux in ('single','residual','variance','matched_count'):
        control='ridge_control_'+aux
        contrast += [{'name':aux+'_minus_base','left':control,'right':base,'primary':False},
                     {'name':'all_at_base_penalty_minus_'+aux,'left':at_base,'right':control,'primary':False}]
    for c in contrast:c['resource']=resource
    seen={}
    for c in contrast:
        pair=(c['left'],c['right'])
        c['alias_of']=seen.get(pair);seen.setdefault(pair,c['name'])
    return contrast


def exact_contrast(left, right, left_predictions, right_predictions, gold, candidates):
    delta=left-right;d=delta[:,0]
    gains=losses=ranking_only=identical=0
    for q,(a,b) in enumerate(zip(left_predictions,right_predictions)):
        la={candidates[i] for i in a if i>=0}&set(gold[q]);lb={candidates[i] for i in b if i>=0}&set(gold[q])
        gains+=len(la-lb);losses+=len(lb-la)
        ranking_only+=int(la==lb and abs(d[q])>1e-12)
        identical+=int(np.array_equal(a,b))
    return {'queries':len(left),'difference':dict(zip(METRICS,delta.mean(axis=0).tolist())),
        'hits_gained_per_100_lists':float(100*delta[:,1].mean()),
        'total_hit_difference':float(delta[:,1].sum()),'supplied_labels_gained':gains,
        'supplied_labels_lost':losses,'ap_wins':int((d>1e-12).sum()),
        'ap_losses':int((d< -1e-12).sum()),'ap_ties':int((abs(d)<=1e-12).sum()),
        'identical_lists':identical,'ranking_only_ap_changes':ranking_only,
        'ap_difference_quantiles':dict(zip(('0','.025','.25','.5','.75','.975','1'),
            np.quantile(d,[0,.025,.25,.5,.75,.975,1],method='linear').tolist()))}


def intervals(draws):
    a=np.asarray(draws)
    return {'nominal_95_percentile':np.quantile(a,[.025,.975],axis=0,method='linear').tolist(),
            'bonferroni_primary_97_5_percentile':np.quantile(a,[.0125,.9875],axis=0,method='linear').tolist()}


def final_statistics(resources, words, categories, gold, spec_by_resource, vocabulary,
                     *, replicates=100000, seed=2026092605, sink=None):
    """Shared draws for all-query AP contrasts and every primary paired metric.

    Secondary contrasts have AP intervals only; all other named metrics retain
    their exact effects. No secondary interval gets a simultaneous-coverage or
    confirmatory claim. `sink` saves draws rather than returning all of them.
    """
    names=('glove','fasttext')
    common=np.logical_and.reduce([resources[n]['usable'] for n in names])
    result={'metric_names':list(METRICS),'primary_resources':list(names),'resources':{},
            'bootstrap':{'replicates':replicates,'seed':seed,'generator':'PCG64',
                         'numpy':np.__version__,'batch_size':500,'quantile':'linear',
                         'stratification':'lexical category-membership signature; recomputed within each panel',
                         'strata':[{'signature':list(sig),'lexical_groups':len(groups),'rows':sum(map(len,groups))}
                                   for sig,groups in membership_groups(words,categories)],
                         'mixed_singleton_limit':'Conditions on the one observed mixed-category lexeme; no between-lexeme uncertainty for that signature'}}
    columns=[];descriptors=[]
    for name in names:
        data=resources[name];ix={s:i for i,s in enumerate(data['systems'])}
        score=data['scores'];panel=masks(categories,data['usable'],common)
        report={'panels':{},'contrasts':{},'aliases':data['aliases']}
        ceilings=ceiling_records(words,gold,data['candidates'],data['counts'],vocabulary)
        if sink:sink(name+'_ceilings',ceilings)
        for pn,mask in panel.items():
            if not mask.any():report['panels'][pn]={'queries':0,'status':'NOT_ESTIMABLE'};continue
            report['panels'][pn]={'queries':int(mask.sum()),'systems':{
                s:dict(zip(METRICS,score[j,mask].mean(axis=0).tolist()))|{'total_hits':int(score[j,mask,1].sum())}
                for j,s in enumerate(data['systems'])},
                'candidate_ceiling':{k:float(np.mean([r[k] for i,r in enumerate(ceilings) if mask[i]]))
                                     for k in ('ap_capped_gold','precision','recall')}}
        contrasts=contrast_specs(name,spec_by_resource[name])
        for c in contrasts:
            left,right=ix[c['left']],ix[c['right']]
            report['contrasts'][c['name']]={**c,**exact_contrast(score[left],score[right],
                data['predictions'][left],data['predictions'][right],gold,data['candidates'])}
            for mi in (range(len(METRICS)) if c['primary'] else (0,)):
                descriptors.append({'resource':name,'contrast':c['name'],'metric':METRICS[mi]})
                columns.append(score[left,:,mi]-score[right,:,mi])
        # All direct systems and prefix controls retain descriptive effects.
        baseline=ix[spec_by_resource[name]['final_selections']['B0_family']]
        report['descriptive_difference_from_base']={s:dict(zip(METRICS,(score[j]-score[baseline]).mean(axis=0).tolist()))
            for j,s in enumerate(data['systems'])}
        countmap=dict(zip(data['candidates'],data['counts']));bands=('0','1','2-5','6+')
        denominators={b:sum(frequency_band(countmap.get(x,0))==b for row in gold for x in row) for b in bands}
        report['training_label_frequency']={'supplied_occurrences':denominators,'systems':{}}
        for j,s in enumerate(data['systems']):
            hits={b:0 for b in bands}
            for q,row in enumerate(data['predictions'][j]):
                for cid in row[row>=0]:
                    x=data['candidates'][cid]
                    if x in gold[q]:hits[frequency_band(countmap[x])]+=1
            report['training_label_frequency']['systems'][s]={b:{'hits':hits[b],
                'micro_reference_recall':hits[b]/denominators[b] if denominators[b] else None} for b in bands}
        result['resources'][name]=report
    draws=paired_stratified_bootstrap(np.column_stack(columns),words,categories,replicates=replicates,seed=seed,batch_size=500)
    if sink:sink('all_query_bootstrap',{'descriptors':descriptors,'values':draws})
    ci95=np.quantile(draws,[.025,.975],axis=0,method='linear')
    ci975=np.quantile(draws,[.0125,.9875],axis=0,method='linear')
    for j,d in enumerate(descriptors):
        target=result['resources'][d['resource']]['contrasts'][d['contrast']]
        target.setdefault('nominal_95_intervals',{})[d['metric']]=ci95[:,j].tolist()
        if d['contrast']=='primary' and d['metric']=='ap_capped_gold':
            target['bonferroni_primary_97_5_interval']=ci975[:,j].tolist()
    for ri,name in enumerate(names):
        data=resources[name];ix={s:i for i,s in enumerate(data['systems'])};sel=spec_by_resource[name]['final_selections']
        l,r=ix[sel['B1_all_family']],ix[sel['B0_family']]
        for pi,pn in enumerate(PANELS):
            mask=masks(categories,data['usable'],common)[pn]
            if not mask.any():continue
            indices=np.flatnonzero(mask)
            delta=data['scores'][l,mask]-data['scores'][r,mask]
            local_seed=seed+100+10*ri+pi
            draw=paired_stratified_bootstrap(delta,[words[i] for i in indices],[categories[i] for i in indices],
                    replicates=replicates,seed=local_seed,batch_size=500)
            if sink:sink(name+'_panel_'+str(pi),{'metrics':list(METRICS),'seed':local_seed,'values':draw})
            result['resources'][name]['panels'][pn]['primary_contrast']={
                **exact_contrast(data['scores'][l,mask],data['scores'][r,mask],
                    data['predictions'][l,mask],data['predictions'][r,mask],
                    [gold[i] for i in indices],data['candidates']),
                'nominal_95_intervals':dict(zip(METRICS,np.quantile(draw,[.025,.975],axis=0,method='linear').T.tolist())),
                'seed':local_seed}
    return result
