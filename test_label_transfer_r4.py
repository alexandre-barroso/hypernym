"""Independent synthetic checks for typed evidence and nested matched selection."""
from collections import Counter
from dataclasses import replace
import math
import unittest
import numpy as np

from label_transfer import Query, make_context, prepare_views, RidgeMoments
from label_transfer_r4 import (
    ARMS, CONFIGS, DIRECT, FEATURES, LAMBDAS, RIDGE_NAMES,
    build_features, capped_ap, choose_candidates, coarse_schedule, crossfit_roles,
    fit_candidates, model_record, predict_candidates, responses,
)
from retrieval_metrics import evaluate


class TypedEvidenceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(681)
        self.matrix = rng.normal(size=(37, 12))
        self.queries = [Query(f"r{i}", f"reference_{i:02}", "Concept" if i % 3 else "Entity",
            tuple(dict.fromkeys(("animal" if i % 2 else "artifact", "living thing" if i % 5 else "place")))) for i in range(35)]
        self.queries += [Query("q0", "query_c", "Concept", ("animal",)), Query("q1", "query_e", "Entity", ("place",))]
        self.vocab = {label for q in self.queries for label in q.labels}
        self.available = np.ones(37, bool)
        self.views = prepare_views(self.matrix, self.available, mean=np.zeros(12), components=np.eye(12), dimensions=tuple(range(2, 13)))
        self.context = make_context(self.queries, range(35), self.vocab)

    def test_features_match_independent_per_label_neighbor_loops(self):
        got = build_features(self.views, self.queries, (35, 36), self.context)
        expected = np.zeros_like(got.features)
        dimensions = [list(range(2, 13)), sorted(set(2 + math.floor(10 * j / 9 + .5) for j in range(10)))]
        for row, qi in enumerate((35, 36)):
            category = self.queries[qi].category
            similarity = {ri: float(self.matrix[qi] @ self.matrix[ri] /
                                     (np.linalg.norm(self.matrix[qi]) * np.linalg.norm(self.matrix[ri]))) for ri in range(35)}
            order = sorted(range(35), key=lambda ri: (-similarity[ri], self.queries[ri].word, self.queries[ri].identifier))
            typed = [ri for ri in order if self.queries[ri].category == category]
            for column, label in enumerate(self.context.candidates):
                support = [ri for ri in range(35) if label in self.queries[ri].labels]
                typed_support = [ri for ri in support if self.queries[ri].category == category]
                expected[row, column, :6] = [len(support) / 35, len(typed_support) / len(typed),
                    max((similarity[ri] for ri in support), default=-1), bool(support),
                    max((similarity[ri] for ri in typed_support), default=-1), bool(typed_support)]
                for offset, pool in ((6, order), (9, typed)):
                    for position, k in enumerate((5, 10, 25)):
                        expected[row, column, offset + position] = sum(label in self.queries[ri].labels for ri in pool[:k]) / min(k, len(pool))
            frequencies = Counter(label for query in self.queries[:35] for label in query.labels)
            for column, schedule in enumerate(dimensions, 12):
                for dimension in schedule:
                    def key(ri):
                        x, y = self.matrix[qi, :dimension], self.matrix[ri, :dimension]
                        return (-float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))), self.queries[ri].word, self.queries[ri].identifier)
                    nearest = min(range(35), key=key)
                    labels = sorted(self.queries[nearest].labels, key=lambda word: (-frequencies[word], word))[:15]
                    for label in labels:
                        expected[row, self.context.lookup[label], column] += 1 / len(schedule)
        np.testing.assert_allclose(got.features, expected, atol=1e-14, rtol=0)

    def test_validation_label_mutations_change_neither_features_nor_context(self):
        first = build_features(self.views, self.queries, (35, 36), self.context)
        changed = list(self.queries)
        changed[35] = replace(changed[35], labels=("new unseen answer",))
        changed[36] = replace(changed[36], labels=("another unseen answer",))
        context = make_context(changed, range(35), self.vocab)
        second = build_features(self.views, changed, (35, 36), context)
        np.testing.assert_array_equal(first.features, second.features)
        self.assertEqual(context.candidates, self.context.candidates)
        self.assertFalse(np.array_equal(responses(self.queries, first, context), responses(changed, second, context)))

    def test_no_same_type_reference_preserves_global_candidates_and_uses_global_prior(self):
        refs = [i for i in range(35) if self.queries[i].category == "Concept"]
        context = make_context(self.queries, refs, self.vocab)
        batch = build_features(self.views, self.queries, (36,), context)
        np.testing.assert_array_equal(batch.features[0, :, 0], batch.features[0, :, 1])
        np.testing.assert_array_equal(batch.features[0, :, 4], -np.ones(len(context.candidates)))
        np.testing.assert_array_equal(batch.features[0, :, 5], np.zeros(len(context.candidates)))
        self.assertFalse(batch.features[0, :, 9:12].any())
        self.assertTrue(batch.eligible[0])

    def test_missing_query_uses_identical_type_prior_fallback(self):
        available = self.available.copy(); available[35] = False
        views = prepare_views(self.matrix, available, mean=np.zeros(12), components=np.eye(12), dimensions=tuple(range(2, 13)))
        batch = build_features(views, self.queries, (35, 36), self.context)
        moments = RidgeMoments(features=14)
        moments.add(batch.features, responses(self.queries, batch, self.context), batch.allowed, batch.eligible)
        fitted, _ = fit_candidates(moments)
        predictions, _ = predict_candidates(batch, self.context, fitted)
        for prediction in predictions[0]:
            np.testing.assert_array_equal(prediction, predictions[0, 0])
        self.assertFalse(batch.features[0, :, 12:].any())

    def test_self_candidate_cannot_enter_predictions_or_recurrence(self):
        queries = list(self.queries); queries[0] = replace(queries[0], labels=("query_c", "animal"))
        context = make_context(queries, range(35), self.vocab | {"query_c"})
        batch = build_features(self.views, queries, (35, 36), context)
        own = context.lookup["query_c"]
        self.assertFalse(batch.allowed[0, own]); self.assertFalse(batch.features[0, own, 12:].any())
        moments = RidgeMoments(features=14); moments.add(batch.features, responses(queries, batch, context), batch.allowed, batch.eligible)
        fitted, _ = fit_candidates(moments); predictions, _ = predict_candidates(batch, context, fitted)
        self.assertNotIn(own, predictions[0])

    def test_exact_ties_zero_prefix_and_dimension_deduplication(self):
        queries = [Query("z", "zulu", "Concept", ("second",)), Query("a", "alpha", "Concept", ("first",)), Query("q", "query", "Concept", ("first",))]
        views = prepare_views([[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 2, 0]], np.ones(3, bool), mean=np.zeros(4), components=np.eye(4), dimensions=(2, 3, 4))
        context = make_context(queries, (0, 1), {"first", "second"})
        batch = build_features(views, queries, (2,), context)
        np.testing.assert_array_equal(batch.neighbors_all, [[-1, 1, 1]])
        np.testing.assert_array_equal(batch.ties_all, [[0, 2, 2]])
        self.assertEqual(batch.features[0, context.lookup["first"], 12], 2 / 3)
        self.assertEqual(coarse_schedule(4), (2, 3, 4))
        self.assertEqual(coarse_schedule(2), (2,))
        self.assertEqual(coarse_schedule(50), (2, 7, 13, 18, 23, 29, 34, 39, 45, 50))
        self.assertEqual(coarse_schedule(300), (2, 35, 68, 101, 134, 168, 201, 234, 267, 300))

    def test_metric_matches_existing_independent_rank_metric(self):
        batch = build_features(self.views, self.queries, (35, 36), self.context)
        moments = RidgeMoments(features=14); moments.add(batch.features, responses(self.queries, batch, self.context), batch.allowed, batch.eligible)
        fitted, _ = fit_candidates(moments); predictions, _ = predict_candidates(batch, self.context, fitted)
        scores = capped_ap(predictions, batch.query_rows, self.context.candidates, self.queries)
        for row, index in enumerate(batch.query_rows):
            for column, _ in enumerate(CONFIGS):
                names = [self.context.candidates[i] for i in predictions[row, column] if i >= 0]
                self.assertAlmostEqual(scores[row, column], evaluate(names, self.queries[index].labels).ap_capped_gold, places=14)


class NestedAndFitTests(unittest.TestCase):
    def test_forbidden_labels_cannot_enter_nested_model_roles(self):
        queries = [Query(str(i), f"word_{i}", "Concept" if i % 2 else "Entity", ("label",)) for i in range(20)]
        roles = crossfit_roles(queries, range(15), range(15, 20), "fixture")
        self.assertEqual(set(r for role in roles for r in role["feature_rows"]), set(range(15)))
        for role in roles:
            self.assertFalse(set(role["feature_rows"]) & set(role["reference_rows"]))
            self.assertEqual(set(role["feature_rows"]) | set(role["reference_rows"]), set(range(15)))
            self.assertFalse(set(role["reference_rows"]) & set(range(15, 20)))
        queries[15] = replace(queries[15], word="word_0")
        with self.assertRaises(ValueError): crossfit_roles(queries, range(15), range(15, 20), "fixture")

    def test_matched_ridge_at_every_penalty_matches_direct_weighted_solve(self):
        rng = np.random.default_rng(438)
        x = rng.normal(size=(6, 23, 14)); x[:, :, 3] = 1.
        y = rng.integers(0, 2, size=(6, 23)).astype(float)
        allowed = rng.random((6, 23)) > .2; eligible = np.array([1, 1, 1, 1, 0, 1], bool)
        moments = RidgeMoments(features=14); moments.add(x, y, allowed, eligible)
        fitted, _ = fit_candidates(moments)
        values = np.concatenate([x[r, allowed[r]] for r in np.flatnonzero(eligible)])
        targets = np.concatenate([y[r, allowed[r]] for r in np.flatnonzero(eligible)])
        weights = np.concatenate([np.full(allowed[r].sum(), 1 / (eligible.sum() * allowed[r].sum())) for r in np.flatnonzero(eligible)])
        for model in fitted.values():
            z = (values[:, model.feature_indices] - model.mean) / model.scale
            design = np.column_stack((np.ones(len(z)), z))
            penalty = np.column_stack((np.zeros(z.shape[1]), np.sqrt(model.regularization) * np.eye(z.shape[1])))
            direct, *_ = np.linalg.lstsq(np.vstack((np.sqrt(weights[:, None]) * design, penalty)), np.r_[np.sqrt(weights) * targets, np.zeros(z.shape[1])], rcond=None)
            np.testing.assert_allclose(np.r_[model.intercept, model.coefficient], direct, rtol=1e-8, atol=1e-9)
            self.assertEqual(len(model_record(model)["features"]), len(model.feature_indices))

    def test_equal_selection_budgets_and_declared_tie_rule(self):
        self.assertEqual(len(DIRECT), 6); self.assertEqual(len(CONFIGS), 21)
        scores = np.zeros(21)
        for arm in ARMS:
            self.assertEqual(len(DIRECT + RIDGE_NAMES[arm]), 11)
        chosen = choose_candidates(scores)
        for arm in ARMS:
            self.assertEqual(chosen[arm + "_family"], DIRECT[0])
            self.assertEqual(chosen[arm + "_ridge"], RIDGE_NAMES[arm][0])
        scores[CONFIGS.index(RIDGE_NAMES["B1_all"][2])] = .2
        chosen = choose_candidates(scores)
        self.assertEqual(chosen["B1_all_family"], RIDGE_NAMES["B1_all"][2])
        self.assertEqual(chosen["B0_family"], DIRECT[0])
        scores[CONFIGS.index(DIRECT[0])] = .2 - 5e-13
        self.assertEqual(choose_candidates(scores)["B1_all_family"], DIRECT[0])


if __name__ == "__main__":
    unittest.main()
