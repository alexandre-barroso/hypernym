"""Task-aware matched ranking with typed evidence and explicit selection grids.

Original numeric code for the separately frozen R4 development protocol. Labels
are read only through a supplied reference context or separate response builder.
The incomplete-list target and recurrence fractions are not semantic probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np

from label_transfer import RidgeMoments, lexical_folds, rank_scores


FEATURES = (
    "prevalence_global", "prevalence_query_type",
    "maximum_support_cosine_global", "has_support_global",
    "maximum_support_cosine_query_type", "has_support_query_type",
    "vote_global_5", "vote_global_10", "vote_global_25",
    "vote_type_5", "vote_type_10", "vote_type_25",
    "recurrence_all", "recurrence_coarse",
)
LAMBDAS = (1., .1, .01, .0001, .000001)
ARMS = {"B0": tuple(range(12)), "B1_all": tuple(range(13)),
        "B1_coarse": tuple(range(12)) + (13,)}
DIRECT = tuple(FEATURES[6:12])
RIDGE_NAMES = {arm: tuple(f"ridge_{arm}_g{i}" for i in range(len(LAMBDAS))) for arm in ARMS}
CONFIGS = DIRECT + tuple(name for arm in ARMS for name in RIDGE_NAMES[arm])


def crossfit_roles(queries, model_rows, forbidden_rows, seed):
    model_rows, forbidden_rows = tuple(model_rows), tuple(forbidden_rows)
    model_types = {queries[row].word for row in model_rows}
    if model_types & {queries[row].word for row in forbidden_rows}:
        raise ValueError("Forbidden validation lexical type entered model-fitting rows")
    assignment = lexical_folds(queries, model_rows, folds=3, seed=seed)
    roles = []
    for fold in range(3):
        feature_rows = [row for row in model_rows if assignment[row] == fold]
        reference_rows = [row for row in model_rows if assignment[row] != fold]
        if not feature_rows or not reference_rows:
            raise ValueError("Need nonempty nested cross-fit roles")
        if {queries[row].word for row in feature_rows} & {queries[row].word for row in reference_rows}:
            raise ValueError("Nested cross-fit lexical overlap")
        roles.append({"crossfit": fold, "feature_rows": feature_rows, "reference_rows": reference_rows})
    return roles


def coarse_schedule(dimension):
    if isinstance(dimension, bool) or not isinstance(dimension, (int, np.integer)) or dimension < 2:
        raise ValueError("A released dimension of at least two is required")
    # Exact integer form of 2 + floor((D-2)*j/9 + 0.5).
    return tuple(sorted({2 + (2 * (int(dimension) - 2) * j + 9) // 18 for j in range(10)}))


def similarity(unit, valid, query_rows, reference_rows):
    scores = np.clip(unit[query_rows] @ unit[reference_rows].T, -1., 1.)
    scores[:, ~valid[reference_rows]] = -np.inf
    scores[~valid[query_rows], :] = -np.inf
    return scores


def recurrence(views, query_rows, context, allowed, dimensions):
    lookup = {dimension: (unit, valid) for dimension, unit, valid in views.prefixes}
    if not dimensions or any(d not in lookup for d in dimensions):
        raise ValueError("Every scheduled prefix must have a supplied numeric view")
    counts = np.zeros(allowed.shape, dtype=np.int64)
    neighbors = np.full((len(query_rows), len(dimensions)), -1, dtype=np.int64)
    ties = np.zeros_like(neighbors)
    for column, dimension in enumerate(dimensions):
        unit, valid = lookup[dimension]
        scores = similarity(unit, valid, query_rows, context.reference_indices)
        chosen = np.argmax(scores, axis=1)
        maxima = scores[np.arange(len(query_rows)), chosen]
        exists = np.isfinite(maxima)
        neighbors[exists, column] = context.reference_indices[chosen[exists]]
        ties[exists, column] = (scores[exists] == maxima[exists, None]).sum(axis=1)
        for row in np.flatnonzero(exists):
            output = [candidate for candidate in context.ordered_labels[chosen[row]]
                      if allowed[row, candidate]][:15]
            counts[row, output] += 1
    return counts / len(dimensions), neighbors, ties


@dataclass
class Batch:
    query_rows: np.ndarray
    features: np.ndarray
    allowed: np.ndarray
    eligible: np.ndarray
    neighbors_all: np.ndarray
    neighbors_coarse: np.ndarray
    ties_all: np.ndarray
    ties_coarse: np.ndarray
    timing: dict


def build_features(views, queries, query_rows, context):
    query_rows = np.asarray(tuple(query_rows), dtype=np.int64)
    if not len(query_rows) or len(set(query_rows.tolist())) != len(query_rows):
        raise ValueError("Need distinct nonempty feature-query rows")
    if any(queries[row].word in context.reference_words for row in query_rows):
        raise ValueError("Query/reference lexical-type leakage")
    nq, nc = len(query_rows), len(context.candidates)
    allowed = np.ones((nq, nc), bool)
    for row, index in enumerate(query_rows):
        own = context.lookup.get(queries[index].word)
        if own is not None:
            allowed[row, own] = False
    x = np.zeros((nq, nc, len(FEATURES)), dtype=np.float64)
    start = time.perf_counter()
    scores = similarity(views.centered_unit, views.centered_valid, query_rows, context.reference_indices)
    ordered = np.argsort(-scores, axis=1, kind="stable")
    retrieval_seconds = time.perf_counter() - start
    valid_ref = views.centered_valid[context.reference_indices]
    eligible = views.centered_valid[query_rows] & valid_ref.any() & allowed.any(axis=1)
    types = np.asarray([queries[row].category for row in context.reference_indices])
    query_types = np.asarray([queries[row].category for row in query_rows])
    start = time.perf_counter()
    x[:, :, 0] = context.counts / len(context.reference_indices)
    for category in ("Concept", "Entity"):
        group = types == category
        rows = np.flatnonzero(query_types == category)
        if len(rows):
            prior = (context.incidence[group].mean(axis=0) if group.any()
                     else context.counts / len(context.reference_indices))
            x[rows, :, 1] = prior
    prior_seconds = time.perf_counter() - start
    start = time.perf_counter()
    for column in range(nc):
        support = np.flatnonzero(context.incidence[:, column] & valid_ref)
        if len(support):
            maxima = scores[:, support].max(axis=1)
            x[:, column, 2] = np.where(np.isfinite(maxima), maxima, -1.)
            x[:, column, 3] = 1.
        else:
            x[:, column, 2] = -1.
        for category in ("Concept", "Entity"):
            rows = np.flatnonzero(query_types == category)
            local_support = support[types[support] == category]
            if len(local_support):
                maxima = scores[np.ix_(rows, local_support)].max(axis=1)
                x[rows, column, 4] = np.where(np.isfinite(maxima), maxima, -1.)
                x[rows, column, 5] = 1.
            else:
                x[rows, column, 4] = -1.
    support_seconds = time.perf_counter() - start
    start = time.perf_counter()
    for row in np.flatnonzero(eligible):
        global_order = ordered[row, :int(valid_ref.sum())]
        typed_order = global_order[types[global_order] == query_types[row]]
        for offset, pool in ((6, global_order), (9, typed_order)):
            for number, k in enumerate((5, 10, 25)):
                if len(pool):
                    x[row, :, offset + number] = context.incidence[pool[:k]].mean(axis=0)
    vote_seconds = time.perf_counter() - start
    all_dimensions = tuple(d for d, _, _ in views.prefixes)
    sparse_dimensions = coarse_schedule(views.centered_unit.shape[1])
    start = time.perf_counter()
    x[:, :, 12], all_neighbors, all_ties = recurrence(views, query_rows, context, allowed, all_dimensions)
    all_seconds = time.perf_counter() - start
    start = time.perf_counter()
    x[:, :, 13], sparse_neighbors, sparse_ties = recurrence(views, query_rows, context, allowed, sparse_dimensions)
    sparse_seconds = time.perf_counter() - start
    return Batch(query_rows, x, allowed, eligible, all_neighbors, sparse_neighbors, all_ties, sparse_ties,
                 {"shared_centered_retrieval": retrieval_seconds, "shared_priors": prior_seconds,
                  "support_features": support_seconds, "vote_features": vote_seconds,
                  "recurrence_all": all_seconds, "recurrence_coarse": sparse_seconds})


def responses(queries, batch, context):
    target = np.zeros(batch.allowed.shape)
    for row, index in enumerate(batch.query_rows):
        for label in queries[index].labels:
            column = context.lookup.get(label)
            if column is not None and batch.allowed[row, column]:
                target[row, column] = 1.
    return target


def fit_candidates(moments):
    fitted, seconds = {}, {}
    for arm, columns in ARMS.items():
        for name, penalty in zip(RIDGE_NAMES[arm], LAMBDAS):
            start = time.perf_counter()
            fitted[name] = moments.fit(feature_indices=columns, regularization=penalty)
            seconds[name] = time.perf_counter() - start
    reference = fitted[RIDGE_NAMES["B0"][0]]
    for model in fitted.values():
        if (not np.array_equal(model.mean[:12], reference.mean)
                or not np.array_equal(model.scale[:12], reference.scale)
                or model.fit_queries != reference.fit_queries):
            raise ValueError("Matched candidate arms have different base scaling or fitting data")
    return fitted, seconds


def model_record(model):
    return {"features": [FEATURES[i] for i in model.feature_indices],
            "feature_indices": list(model.feature_indices), "mean": model.mean.tolist(),
            "scale": model.scale.tolist(), "coefficient": model.coefficient.tolist(),
            "intercept": model.intercept, "regularization": model.regularization,
            "fit_queries": model.fit_queries, "condition_number": model.condition_number}


def predict_candidates(batch, context, fitted, depth=15):
    result = np.full((len(batch.query_rows), len(CONFIGS), depth), -1, dtype=np.int32)
    times = {}
    for column, name in enumerate(CONFIGS):
        start = time.perf_counter()
        scores = (batch.features[:, :, FEATURES.index(name)] if name in DIRECT
                  else fitted[name].predict(batch.features))
        for row in range(len(batch.query_rows)):
            primary = scores[row] if batch.eligible[row] else batch.features[row, :, 1]
            ranking = rank_scores(primary, batch.allowed[row], context.counts,
                                  secondary=batch.features[row, :, 1], depth=depth)
            result[row, column, :len(ranking)] = ranking
        times[name] = time.perf_counter() - start
    return result, times


def capped_ap(predictions, query_rows, candidates, queries):
    """Per-query/config supplied-list AP; scalar metric audit remains separate."""
    result = np.zeros(predictions.shape[:2])
    for row, index in enumerate(query_rows):
        labels = set(queries[index].labels)
        for config in range(predictions.shape[1]):
            hits = 0
            numerator = 0.
            for rank, candidate in enumerate(predictions[row, config], 1):
                if candidate >= 0 and candidates[int(candidate)] in labels:
                    hits += 1
                    numerator += hits / rank
            result[row, config] = numerator / min(predictions.shape[-1], len(labels))
    return result


def choose_candidates(mean_ap):
    if np.asarray(mean_ap).shape != (len(CONFIGS),) or not np.isfinite(mean_ap).all():
        raise ValueError("Every fixed candidate needs a finite tuning score")
    selected = {}
    for arm in ARMS:
        for scope, names in (("family", DIRECT + RIDGE_NAMES[arm]), ("ridge", RIDGE_NAMES[arm])):
            best = names[0]
            for name in names[1:]:
                if mean_ap[CONFIGS.index(name)] > mean_ap[CONFIGS.index(best)] + 1e-12:
                    best = name
            selected[arm + "_" + scope] = best
    return selected
