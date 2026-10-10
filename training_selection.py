"""Replay the training-only nested comparison and complete prefix selection.

Uses the original numerical functions and deterministic folds. Reads the pinned
training members and the independently reproduced vector cache; never opens test
references. This replay is not a new prospective experiment.
"""
from pathlib import Path
import argparse, json, os, time
import numpy as np
from pipeline import dataset, read, save, SPEC, digest, INPUTS
from selection_folds import fold_manifest
from label_transfer import make_context, prepare_views, RidgeMoments, rank_scores
from label_transfer_r4 import (FEATURES, CONFIGS, crossfit_roles, build_features,
    responses, fit_candidates, predict_candidates, capped_ap, choose_candidates)

def build(queries, vocabulary, views, rows, refs):
    context = make_context(queries, refs, vocabulary)
    return context, build_features(views, queries, rows, context)

def prefix_scores(batch, context, queries, views):
    predictions = np.full((len(batch.query_rows), len(views.prefixes), 15), -1, dtype=np.int32)
    positions = {int(row): i for i, row in enumerate(context.reference_indices)}
    for row, query in enumerate(batch.query_rows):
        for column, neighbor in enumerate(batch.neighbors_all[row]):
            if not views.available[query]:
                labels = rank_scores(context.counts.astype(float), batch.allowed[row], context.counts)
            elif neighbor < 0:
                labels = ()
            else:
                labels = tuple(i for i in context.ordered_labels[positions[int(neighbor)]] if batch.allowed[row, i])[:15]
            predictions[row, column, :len(labels)] = labels
    return capped_ap(predictions, batch.query_rows, context.candidates, queries)

def run(a):
    queries, vocabulary = dataset(a.dataset_zip, 'train')
    spec = SPEC[a.resource]; directory = a.run/a.resource
    descriptor = read(directory/'vectors.json')
    expected_records = [dict(identifier=q.identifier, word=q.word, category=q.category) for q in queries]
    assert descriptor['records'][:1500] == expected_records
    assert descriptor['archive_sha256'] == INPUTS[a.resource]['sha256']
    with np.load(directory/'vectors.npz', allow_pickle=False) as z:
        views = prepare_views(z['matrix'][:1500], z['available'][:1500],
            mean=z['mean'], components=z['components'], dimensions=tuple(spec['all_prefix_schedule']))
    manifest = fold_manifest(queries)
    out = a.output/a.resource; out.mkdir(parents=True, exist_ok=False)
    save(out/'folds.json', manifest)
    start = time.perf_counter(); sums = np.zeros(len(CONFIGS)); count = 0
    prefix_sums = np.zeros(len(views.prefixes)); outer_records = []
    for fold in manifest['folds']:
        outer = fold['outer_fold']; tuning = np.zeros(len(CONFIGS)); tuning_count = 0
        final_moments = RidgeMoments(len(FEATURES)); middle_records = []
        for middle in fold['inner_folds']:
            seed = f"label-transfer-r4-fit|outer={outer}|middle={middle['inner_fold']}"
            roles = crossfit_roles(queries, middle['reference_rows'],
                middle['feature_query_rows'] + fold['validation_rows'], seed)
            moments = RidgeMoments(len(FEATURES))
            for role in roles:
                context, batch = build(queries, vocabulary, views, role['feature_rows'], role['reference_rows'])
                moments.add(batch.features, responses(queries, batch, context), batch.allowed, batch.eligible)
            fitted, _ = fit_candidates(moments)
            context, batch = build(queries, vocabulary, views, middle['feature_query_rows'], middle['reference_rows'])
            predictions, _ = predict_candidates(batch, context, fitted)
            scores = capped_ap(predictions, batch.query_rows, context.candidates, queries)
            tuning += scores.sum(axis=0); tuning_count += len(batch.query_rows)
            final_moments.add(batch.features, responses(queries, batch, context), batch.allowed, batch.eligible)
            middle_records.append(dict(middle_fold=middle['inner_fold'], crossfit_roles=roles,
                mean_ap=scores.mean(axis=0).tolist()))
        fitted, _ = fit_candidates(final_moments)
        context, batch = build(queries, vocabulary, views, fold['validation_rows'], fold['training_rows'])
        predictions, _ = predict_candidates(batch, context, fitted)
        scores = capped_ap(predictions, batch.query_rows, context.candidates, queries)
        prefixes = prefix_scores(batch, context, queries, views)
        np.savez_compressed(out/f'outer{outer}.npz', query_rows=batch.query_rows,
            predictions=predictions, ap=scores, prefix_ap=prefixes)
        save(out/f'outer{outer}.json', dict(candidates=list(context.candidates), configs=list(CONFIGS),
            validation_rows=batch.query_rows.tolist(), training_rows=fold['training_rows']))
        sums += scores.sum(axis=0); prefix_sums += prefixes.sum(axis=0); count += len(batch.query_rows)
        outer_records.append(dict(outer_fold=outer, tuning_means=(tuning/tuning_count).tolist(),
            selected=choose_candidates(tuning/tuning_count), middle_folds=middle_records))
        print(a.resource, 'completed outer fold', outer, flush=True)
    assert count == 1500
    means = sums/count; prefix_means = prefix_sums/count
    expected = np.array([spec['candidate_means'][name] for name in CONFIGS])
    expected_prefix = np.array([row['training_oof_ap_capped_gold'] for row in spec['single_prefix']['complete_training_oof_curve']])
    assert np.allclose(means, expected, rtol=0, atol=1e-12), np.max(abs(means-expected))
    assert np.allclose(prefix_means, expected_prefix, rtol=0, atol=1e-12), np.max(abs(prefix_means-expected_prefix))
    selected = choose_candidates(means)
    chosen_dimension = max(d for d, ap in zip(spec['all_prefix_schedule'], prefix_means) if ap >= max(prefix_means)-1e-12)
    assert selected == spec['final_selections']
    assert chosen_dimension == spec['single_prefix']['selected']['dimension']
    result = dict(status='PASS', resource=a.resource, training_queries=count,
        candidate_means=dict(zip(CONFIGS, means.tolist())), final_selections=selected,
        single_prefix_dimension=chosen_dimension, complete_prefix_ap=prefix_means.tolist(),
        maximum_mean_difference=float(np.max(abs(means-expected))),
        maximum_prefix_difference=float(np.max(abs(prefix_means-expected_prefix))),
        outer_folds=outer_records, elapsed_seconds=time.perf_counter()-start,
        vector_cache_sha256=digest(directory/'vectors.npz'), dataset_sha256=INPUTS['dataset']['sha256'],
        scope='Numerical replay of the original training-only selection; not a new prospective freeze or timing comparison.')
    save(out/'verification.json', result)
    print('PASS', a.resource, 'training selections and complete prefix curve', flush=True)

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resource', choices=list(SPEC), required=True)
    p.add_argument('--dataset-zip', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True, help='Vector caches created by pipeline.py vectors')
    p.add_argument('--output', type=Path, required=True, help='New directory for the selection replay')
    a=p.parse_args()
    for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
        if os.environ.get(key) != '1':p.error('Launch with '+key+'=1')
    run(a)
