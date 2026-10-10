"""Original task-aware label-transfer and dimensional-recurrence primitives.

Reference-list agreement is the supervised target. Regression scores and prefix
fractions are not probabilities of linguistic truth. No file, model object or
foreign implementation is loaded here. Only NumPy and the standard library are
required. See the development protocol for the specific experiment's choices.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import time
from typing import Sequence

import numpy as np


BASE_FEATURE_NAMES = (
    "label_prevalence", "maximum_centered_support_cosine",
    "centered_top10_label_support_fraction", "has_centered_reference_support",
)
FEATURE_NAMES = BASE_FEATURE_NAMES + ("dimensional_label_recurrence",)


@dataclass(frozen=True)
class Query:
    identifier: str
    word: str
    category: str
    labels: tuple[str, ...]

    def __post_init__(self):
        if (not isinstance(self.identifier, str) or not self.identifier
                or not isinstance(self.word, str) or not self.word or self.word != self.word.lower()):
            raise ValueError("Query IDs and lowercase lexical types are required")
        if self.category not in ("Concept", "Entity"):
            raise ValueError("Expected the official Concept/Entity stratum")
        if (not self.labels or len(set(self.labels)) != len(self.labels)
                or any(not isinstance(x, str) or not x or x != x.lower() for x in self.labels)):
            raise ValueError("Reference labels must be unique nonempty lowercase strings")


def _norms(rows):
    scale = np.max(np.abs(rows), axis=1)
    unit_scale = np.zeros_like(rows)
    np.divide(rows, scale[:, None], out=unit_scale, where=scale[:, None] != 0)
    return scale * np.sqrt(np.einsum("ij,ij->i", unit_scale, unit_scale))


def _unit(rows, available):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        norm = _norms(rows)
        valid = available & (norm > 0)
        unit = np.zeros_like(rows)
        np.divide(rows, norm[:, None], out=unit, where=valid[:, None])
    return unit, valid


@dataclass(frozen=True)
class Views:
    raw_unit: np.ndarray
    raw_valid: np.ndarray
    centered_unit: np.ndarray
    centered_valid: np.ndarray
    prefixes: tuple[tuple[int, np.ndarray, np.ndarray], ...]
    available: np.ndarray


def prepare_views(matrix, available, *, mean, components, dimensions) -> Views:
    values = np.asarray(matrix)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1] or values.dtype.kind not in "iuf":
        raise ValueError("Expected a nonempty real numeric query matrix")
    values = np.array(values, dtype=np.float64, copy=True)
    available = np.asarray(available)
    if available.dtype.kind != "b" or available.shape != (len(values),):
        raise ValueError("Expected a boolean availability vector")
    mean = np.asarray(mean, dtype=np.float64)
    components = np.asarray(components, dtype=np.float64)
    d = values.shape[1]
    if mean.shape != (d,) or components.shape != (d, d):
        raise ValueError("Expected a full external PCA mean and column eigenvectors")
    if not all(np.isfinite(x).all() for x in (values, mean, components)):
        raise ValueError("Numeric inputs must be finite")
    if not np.allclose(components.T @ components, np.eye(d), rtol=0, atol=1e-8):
        raise ValueError("External PCA columns must be orthonormal")
    dimensions = tuple(dimensions)
    if (not dimensions or any(isinstance(k, bool) or not isinstance(k, (int, np.integer))
                             or k < 1 or k > d for k in dimensions)
            or len(set(dimensions)) != len(dimensions)):
        raise ValueError("Prefix dimensions must be distinct integers in [1,d]")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        coordinates = (values - mean) @ components
        raw_unit, raw_valid = _unit(values, available)
        centered_unit, centered_valid = _unit(coordinates, available)
        prefixes = tuple((int(k), *_unit(coordinates[:, :k], available)) for k in dimensions)
    return Views(raw_unit, raw_valid, centered_unit, centered_valid, prefixes, available.copy())


def lexical_folds(queries: Sequence[Query], indices, *, folds: int, seed: str):
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ValueError("Need at least two folds")
    indices = tuple(int(i) for i in indices)
    if len(set(indices)) != len(indices):
        raise ValueError("Repeated row index")
    groups = {}
    for index in indices:
        query = queries[index]
        if query.word in groups and queries[groups[query.word][0]].category != query.category:
            raise ValueError("One lexical type occurs in conflicting strata")
        groups.setdefault(query.word, []).append(index)
    assignments = {}
    for category in ("Concept", "Entity"):
        words = [word for word, rows in groups.items() if queries[rows[0]].category == category]
        words.sort(key=lambda word: (hashlib.sha256(
            (seed + "|" + category + "|" + word).encode()).hexdigest(), word))
        for position, word in enumerate(words):
            for index in groups[word]:
                assignments[index] = position % folds
    return assignments


@dataclass(frozen=True)
class LabelContext:
    reference_indices: np.ndarray
    reference_words: frozenset[str]
    candidates: tuple[str, ...]
    lookup: dict[str, int]
    counts: np.ndarray
    incidence: np.ndarray
    ordered_labels: tuple[tuple[int, ...], ...]


def make_context(queries: Sequence[Query], reference_indices, vocabulary) -> LabelContext:
    reference_indices = tuple(int(i) for i in reference_indices)
    if not reference_indices or len(set(reference_indices)) != len(reference_indices):
        raise ValueError("Need distinct nonempty reference rows")
    reference_indices = tuple(sorted(reference_indices,
        key=lambda i: (queries[i].word, queries[i].identifier)))
    counts = Counter(label for i in reference_indices for label in queries[i].labels if label in vocabulary)
    candidates = tuple(sorted(counts))
    if not candidates:
        raise ValueError("Reference dictionary has no official candidate labels")
    lookup = {word: i for i, word in enumerate(candidates)}
    frequencies = np.asarray([counts[word] for word in candidates], dtype=np.int64)
    incidence = np.zeros((len(reference_indices), len(candidates)), dtype=bool)
    ordered = []
    for position, index in enumerate(reference_indices):
        local = [lookup[label] for label in queries[index].labels if label in lookup]
        incidence[position, local] = True
        ordered.append(tuple(sorted(local, key=lambda j: (-frequencies[j], candidates[j]))))
    return LabelContext(np.asarray(reference_indices, dtype=np.int64),
        frozenset(queries[i].word for i in reference_indices), candidates, lookup,
        frequencies, incidence, tuple(ordered))


def _similarities(unit, valid, query_indices, reference_indices):
    scores = np.clip(unit[query_indices] @ unit[reference_indices].T, -1.0, 1.0)
    scores[:, ~valid[reference_indices]] = -np.inf
    scores[~valid[query_indices], :] = -np.inf
    return scores


def rank_scores(scores, allowed, prior, *, depth=15, secondary=None):
    """Descending score, optional descending secondary, prior, then lexical ID.

    Candidate rows are already lexically sorted by make_context. This helper
    excludes disallowed/nonfinite scores and returns integer candidate positions.
    """
    pool = np.flatnonzero(allowed & np.isfinite(scores))
    if secondary is None:
        secondary = np.zeros_like(scores)
    order = np.lexsort((pool, -prior[pool], -secondary[pool], -scores[pool]))
    return tuple(int(i) for i in pool[order[:depth]])


def _transfer(context, neighbor_order, allowed, *, k, depth=15):
    selected = []
    seen = set()
    for position in neighbor_order[:k]:
        for candidate in context.ordered_labels[int(position)]:
            if allowed[candidate] and candidate not in seen:
                selected.append(candidate)
                seen.add(candidate)
                if len(selected) == depth:
                    return tuple(selected)
    return tuple(selected)


@dataclass
class FeatureBatch:
    query_indices: np.ndarray
    features: np.ndarray
    allowed: np.ndarray
    fit_eligible: np.ndarray
    prefix_neighbors: np.ndarray
    prefix_tie_counts: np.ndarray
    prefix_missing: np.ndarray
    prefix_predictions: tuple[tuple[tuple[int, ...], ...], ...] | None
    centered_order: np.ndarray
    centered_scores: np.ndarray
    raw_scores: np.ndarray | None
    timing_seconds: dict


def measure_features(views: Views, queries: Sequence[Query], query_indices,
                     context: LabelContext, *, retain_comparators=False) -> FeatureBatch:
    """Create features without reading the query rows' labels.

    Label access is confined to make_context's explicitly supplied reference
    rows. Repeated query/reference lexical types are rejected, including aliases
    of the same lowercased type with distinct record IDs.
    """
    query_indices = np.asarray(tuple(query_indices), dtype=np.int64)
    if len(set(query_indices.tolist())) != len(query_indices):
        raise ValueError("Repeated query row")
    if any(queries[i].word in context.reference_words for i in query_indices):
        raise ValueError("Query/reference lexical-type leakage")
    nq, nc = len(query_indices), len(context.candidates)
    allowed = np.ones((nq, nc), dtype=bool)
    for row, index in enumerate(query_indices):
        own = context.lookup.get(queries[index].word)
        if own is not None:
            allowed[row, own] = False
    features = np.zeros((nq, nc, len(FEATURE_NAMES)), dtype=np.float64)
    start = time.perf_counter()
    centered = _similarities(views.centered_unit, views.centered_valid,
                            query_indices, context.reference_indices)
    centered_order = np.argsort(-centered, axis=1, kind="stable")
    centered_seconds = time.perf_counter() - start
    start = time.perf_counter()
    features[:, :, 0] = context.counts / len(context.reference_indices)
    valid_references = views.centered_valid[context.reference_indices]
    for column in range(nc):
        support = np.flatnonzero(context.incidence[:, column] & valid_references)
        if len(support):
            maxima = np.max(centered[:, support], axis=1)
            features[:, column, 1] = np.where(np.isfinite(maxima), maxima, -1.0)
            features[:, column, 3] = 1.0
        else:
            features[:, column, 1] = -1.0
    k = min(10, int(valid_references.sum()))
    fit_eligible = views.centered_valid[query_indices] & (k > 0) & allowed.any(axis=1)
    if k:
        features[:, :, 2] = context.incidence[centered_order[:, :k]].mean(axis=1)
        features[~fit_eligible, :, 2] = 0.0
    base_seconds = time.perf_counter() - start

    start = time.perf_counter()
    nd = len(views.prefixes)
    counts = np.zeros((nq, nc), dtype=np.int64)
    prefix_neighbors = np.full((nq, nd), -1, dtype=np.int64)
    prefix_ties = np.zeros((nq, nd), dtype=np.int64)
    prefix_missing = np.ones((nq, nd), dtype=bool)
    predictions = [] if retain_comparators else None
    for column, (_, unit, valid) in enumerate(views.prefixes):
        scores = _similarities(unit, valid, query_indices, context.reference_indices)
        nearest = np.argmax(scores, axis=1)
        maxima = scores[np.arange(nq), nearest]
        exists = np.isfinite(maxima)
        prefix_neighbors[exists, column] = context.reference_indices[nearest[exists]]
        prefix_ties[exists, column] = (scores[exists] == maxima[exists, None]).sum(axis=1)
        prefix_missing[:, column] = ~exists
        current = [] if retain_comparators else None
        for row in range(nq):
            output = (_transfer(context, (nearest[row],), allowed[row], k=1)
                      if exists[row] else ())
            counts[row, list(output)] += 1
            if retain_comparators:
                current.append(output)
        if retain_comparators:
            predictions.append(tuple(current))
    features[:, :, 4] = counts / nd
    recurrence_seconds = time.perf_counter() - start
    raw_scores = None
    raw_seconds = 0.0
    if retain_comparators:
        start = time.perf_counter()
        raw_scores = _similarities(views.raw_unit, views.raw_valid,
                                  query_indices, context.reference_indices)
        raw_seconds = time.perf_counter() - start
    return FeatureBatch(query_indices, features, allowed, fit_eligible,
        prefix_neighbors, prefix_ties, prefix_missing,
        tuple(predictions) if retain_comparators else None,
        centered_order, centered, raw_scores,
        {"centered_similarity_and_order": centered_seconds,
         "base_features_after_shared_similarity": base_seconds,
         "recurrence_increment_all_prefixes": recurrence_seconds,
         "raw_similarity": raw_seconds})


def response_matrix(queries, batch: FeatureBatch, context: LabelContext):
    """Supervised target creation, deliberately separate from feature generation."""
    response = np.zeros(batch.allowed.shape, dtype=np.float64)
    for row, index in enumerate(batch.query_indices):
        for label in queries[index].labels:
            column = context.lookup.get(label)
            if column is not None and batch.allowed[row, column]:
                response[row, column] = 1.0
    return response


@dataclass(frozen=True)
class RidgeRanker:
    feature_indices: tuple[int, ...]
    mean: np.ndarray
    scale: np.ndarray
    coefficient: np.ndarray
    intercept: float
    regularization: float
    fit_queries: int
    condition_number: float

    def predict(self, features):
        selected = np.asarray(features)[..., self.feature_indices]
        return ((selected - self.mean) / self.scale) @ self.coefficient + self.intercept

    def record(self):
        return {"feature_names": [FEATURE_NAMES[i] for i in self.feature_indices],
                "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "coefficient": self.coefficient.tolist(), "intercept": self.intercept,
                "regularization": self.regularization, "fit_queries": self.fit_queries,
                "regularized_gram_condition_number": self.condition_number,
                "score_interpretation": "Real ranking score for supplied reference membership, not a semantic probability"}


class RidgeMoments:
    """Sufficient statistics for a query-balanced standardized ridge objective."""

    def __init__(self, features=len(FEATURE_NAMES)):
        self.feature_count = features
        self.queries = 0
        self.candidate_examples = 0
        self.sum_x = np.zeros(features)
        self.sum_xx = np.zeros((features, features))
        self.sum_y = 0.0
        self.sum_xy = np.zeros(features)

    def add(self, features, response, allowed, eligible):
        x, y = np.asarray(features), np.asarray(response)
        allowed, eligible = np.asarray(allowed), np.asarray(eligible)
        if (x.ndim != 3 or x.shape[-1] != self.feature_count or y.shape != x.shape[:2]
                or allowed.shape != y.shape or allowed.dtype.kind != "b"
                or eligible.shape != (len(x),) or eligible.dtype.kind != "b"):
            raise ValueError("Mismatched ridge example shapes")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Ridge inputs must be finite")
        for row in np.flatnonzero(eligible & allowed.any(axis=1)):
            values, targets = x[row, allowed[row]], y[row, allowed[row]]
            self.sum_x += values.mean(axis=0)
            self.sum_xx += (values.T @ values) / len(values)
            self.sum_y += float(targets.mean())
            self.sum_xy += (values.T @ targets) / len(values)
            self.queries += 1
            self.candidate_examples += len(values)

    def fit(self, *, feature_indices, regularization=0.1):
        if not self.queries:
            raise ValueError("No eligible cross-fitted queries")
        indices = tuple(int(i) for i in feature_indices)
        if (not indices or len(set(indices)) != len(indices)
                or min(indices) < 0 or max(indices) >= self.feature_count
                or not np.isfinite(regularization) or regularization <= 0):
            raise ValueError("Invalid ridge features or regularization")
        mean = self.sum_x[list(indices)] / self.queries
        raw_gram = self.sum_xx[np.ix_(indices, indices)] / self.queries
        centered_gram = (raw_gram - np.outer(mean, mean))
        centered_gram = (centered_gram + centered_gram.T) / 2
        variance = np.maximum(np.diag(centered_gram), 0)
        scale = np.sqrt(variance)
        scale[variance <= 64 * np.finfo(np.float64).eps * np.maximum(1, np.diag(raw_gram))] = 1.0
        target_mean = self.sum_y / self.queries
        xy = self.sum_xy[list(indices)] / self.queries - mean * target_mean
        gram = centered_gram / np.outer(scale, scale)
        regularized = gram + regularization * np.eye(len(indices))
        coefficient = np.linalg.solve(regularized, xy / scale)
        return RidgeRanker(indices, mean, scale, coefficient, float(target_mean),
                           float(regularization), self.queries,
                           float(np.linalg.cond(regularized)))


def query_predictions(batch: FeatureBatch, context: LabelContext, views: Views,
                      *, b0: RidgeRanker, b1: RidgeRanker, depth=15):
    """Return all comparator lists in shared candidate-index coordinates."""
    if batch.raw_scores is None or batch.prefix_predictions is None:
        raise ValueError("Comparator data was not retained")
    learned = {}
    model_times = {}
    for name, model in (("B0", b0), ("B1", b1)):
        start = time.perf_counter()
        learned[name] = model.predict(batch.features)
        model_times[name + "_score_matrix"] = time.perf_counter() - start
        model_times[name + "_ranking"] = 0.0
    raw_order = np.argsort(-batch.raw_scores, axis=1, kind="stable")
    outputs = []
    for row, query_index in enumerate(batch.query_indices):
        allowed = batch.allowed[row]
        prior = rank_scores(context.counts.astype(float), allowed, context.counts, depth=depth)
        result = {"label_prior": prior}
        fallback = not views.available[query_index]
        for name, scores, order in (("raw_nearest", batch.raw_scores, raw_order),
                                    ("centered_nearest", batch.centered_scores, batch.centered_order)):
            valid_order = [int(j) for j in order[row] if np.isfinite(scores[row, j])]
            result[name] = (_transfer(context, valid_order, allowed, k=1, depth=depth)
                            if valid_order and not fallback else prior)
        valid_order = [int(j) for j in batch.centered_order[row]
                       if np.isfinite(batch.centered_scores[row, j])]
        for k in (5, 10, 25):
            result[f"ordered_top{k}"] = (_transfer(context, valid_order, allowed, k=k, depth=depth)
                                         if valid_order and not fallback else prior)
            votes = (context.incidence[valid_order[:k]].mean(axis=0)
                     if valid_order and not fallback else context.counts.astype(float))
            result[f"vote_top{k}"] = rank_scores(votes, allowed, context.counts, depth=depth)
        result["recurrence_fixed15"] = (
            rank_scores(batch.features[row, :, 4], allowed, context.counts, depth=depth,
                        secondary=batch.features[row, :, 1])
            if batch.fit_eligible[row] and not fallback else prior)
        for name in ("B0", "B1"):
            start = time.perf_counter()
            result[name] = (rank_scores(learned[name][row], allowed, context.counts, depth=depth)
                            if batch.fit_eligible[row] and not fallback else prior)
            model_times[name + "_ranking"] += time.perf_counter() - start
        for prefix, (dimension, _, _) in enumerate(views.prefixes):
            result[f"prefix_nn_{dimension:03}"] = (prior if fallback else batch.prefix_predictions[prefix][row])
        outputs.append(result)
    return outputs, model_times
