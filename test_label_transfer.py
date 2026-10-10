"""Synthetic checks for original transfer, recurrence and fold-local ranking."""

from collections import Counter
from dataclasses import replace
import unittest

import numpy as np

from label_transfer import (
    Query, RidgeMoments, lexical_folds, make_context, measure_features,
    prepare_views, query_predictions, rank_scores, response_matrix,
)


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.queries = [
            Query("r0", "alpha", "Concept", ("living thing", "organism")),
            Query("r1", "beta", "Concept", ("organism", "plant")),
            Query("r2", "gamma", "Entity", ("person", "living thing")),
            Query("r3", "delta", "Concept", ("artifact",)),
            Query("q0", "epsilon", "Concept", ("organism",)),
            Query("q1", "zeta", "Entity", ("person",)),
        ]
        self.matrix = np.array([[2., 1., 1.], [1., 3., -1.], [-2., 1., 2.],
                                [-2., -3., 1.], [2., 1.2, .9], [-1.5, .8, 1.7]])
        self.vocabulary = {label for q in self.queries for label in q.labels}
        self.views = prepare_views(self.matrix, np.ones(6, dtype=bool),
            mean=np.zeros(3), components=np.eye(3), dimensions=(1, 2, 3))
        self.context = make_context(self.queries, (0, 1, 2, 3), self.vocabulary)

    def test_recurrence_matches_independent_scalar_loop(self):
        got = measure_features(self.views, self.queries, (4, 5), self.context,
                               retain_comparators=True)
        frequencies = Counter(label for q in self.queries[:4] for label in q.labels)
        expected = np.zeros((2, len(self.context.candidates)))
        for row, qi in enumerate((4, 5)):
            for column, dimension in enumerate((1, 2, 3)):
                values = []
                x = self.matrix[qi, :dimension]
                for ri in range(4):
                    y = self.matrix[ri, :dimension]
                    similarity = float(sum(x[j] * y[j] for j in range(dimension)) /
                                       (np.linalg.norm(x) * np.linalg.norm(y)))
                    values.append((-similarity, self.queries[ri].word, self.queries[ri].identifier, ri))
                chosen = min(values)[3]
                self.assertEqual(got.prefix_neighbors[row, column], chosen)
                labels = sorted(self.queries[chosen].labels, key=lambda label: (-frequencies[label], label))
                for label in labels[:15]:
                    expected[row, self.context.lookup[label]] += 1 / 3
        np.testing.assert_allclose(got.features[:, :, 4], expected, rtol=0, atol=1e-15)

    def test_multiword_labels_need_no_candidate_vector(self):
        self.assertNotIn("living thing", {q.word for q in self.queries})
        batch = measure_features(self.views, self.queries, (4,), self.context, retain_comparators=True)
        labels = [self.context.candidates[i] for i in batch.prefix_predictions[-1][0]]
        self.assertIn("living thing", labels)

    def test_changing_query_labels_cannot_change_features(self):
        first = measure_features(self.views, self.queries, (4, 5), self.context)
        changed = list(self.queries)
        changed[4] = replace(changed[4], labels=("entirely new unknown answer",))
        changed[5] = replace(changed[5], labels=("another answer",))
        context = make_context(changed, (0, 1, 2, 3), self.vocabulary)
        second = measure_features(self.views, changed, (4, 5), context)
        np.testing.assert_array_equal(first.features, second.features)
        np.testing.assert_array_equal(first.prefix_neighbors, second.prefix_neighbors)
        self.assertFalse(np.array_equal(response_matrix(self.queries, first, self.context),
                                        response_matrix(changed, second, context)))

    def test_query_reference_type_overlap_rejected(self):
        changed = list(self.queries)
        changed[4] = replace(changed[4], word="alpha")
        with self.assertRaisesRegex(ValueError, "leakage"):
            measure_features(self.views, changed, (4,), self.context)

    def test_exact_cosine_tie_uses_lexical_type_not_input_order(self):
        queries = [Query("r0", "zulu", "Concept", ("last",)),
                   Query("r1", "alpha", "Concept", ("first",)),
                   Query("q", "query", "Concept", ("first",))]
        matrix = np.array([[1., 0.], [1., 0.], [2., 0.]])
        views = prepare_views(matrix, np.ones(3, bool), mean=np.zeros(2),
                              components=np.eye(2), dimensions=(1, 2))
        context = make_context(queries, (0, 1), {"first", "last"})
        batch = measure_features(views, queries, (2,), context)
        np.testing.assert_array_equal(batch.prefix_neighbors, [[1, 1]])
        np.testing.assert_array_equal(batch.prefix_tie_counts, [[2, 2]])

    def test_zero_prefix_abstains_and_stays_in_denominator(self):
        queries = [Query("r", "reference", "Concept", ("target",)),
                   Query("q", "query", "Concept", ("target",))]
        views = prepare_views([[0., 1.], [0., 2.]], np.ones(2, bool),
                              mean=np.zeros(2), components=np.eye(2), dimensions=(1, 2))
        context = make_context(queries, (0,), {"target"})
        batch = measure_features(views, queries, (1,), context)
        np.testing.assert_array_equal(batch.prefix_missing, [[True, False]])
        self.assertEqual(batch.features[0, 0, 4], .5)

    def test_zero_reference_is_not_an_arbitrary_nearest_neighbor(self):
        matrix = self.matrix.copy()
        matrix[0] = 0
        views = prepare_views(matrix, np.ones(6, bool), mean=np.zeros(3),
                              components=np.eye(3), dimensions=(1, 2, 3))
        batch = measure_features(views, self.queries, (4, 5), self.context)
        self.assertFalse(np.any(batch.prefix_neighbors == 0))

    def test_unrepresented_query_is_prior_fallback_in_all_systems(self):
        available = np.ones(6, bool)
        available[4] = False
        views = prepare_views(self.matrix, available, mean=np.zeros(3),
                              components=np.eye(3), dimensions=(1, 2, 3))
        batch = measure_features(views, self.queries, (4, 5), self.context, retain_comparators=True)
        stats = RidgeMoments()
        stats.add(batch.features, response_matrix(self.queries, batch, self.context),
                  batch.allowed, batch.fit_eligible)
        outputs, _ = query_predictions(batch, self.context, views,
            b0=stats.fit(feature_indices=(0, 1, 2, 3)),
            b1=stats.fit(feature_indices=(0, 1, 2, 3, 4)))
        for ranking in outputs[0].values():
            self.assertEqual(ranking, outputs[0]["label_prior"])
        self.assertEqual(stats.queries, 1)

    def test_self_candidate_is_excluded_from_training_and_prediction(self):
        queries = list(self.queries)
        queries[0] = replace(queries[0], labels=("epsilon", "living thing"))
        context = make_context(queries, (0, 1, 2, 3), self.vocabulary | {"epsilon"})
        batch = measure_features(self.views, queries, (4,), context, retain_comparators=True)
        own = context.lookup["epsilon"]
        self.assertFalse(batch.allowed[0, own])
        self.assertEqual(batch.features[0, own, 4], 0)
        self.assertEqual(response_matrix(queries, batch, context)[0, own], 0)

    def test_ordered_transfer_deduplicates_shared_labels(self):
        batch = measure_features(self.views, self.queries, (4, 5), self.context, retain_comparators=True)
        stats = RidgeMoments()
        stats.add(batch.features, response_matrix(self.queries, batch, self.context), batch.allowed, batch.fit_eligible)
        outputs, _ = query_predictions(batch, self.context, self.views,
            b0=stats.fit(feature_indices=(0, 1, 2, 3)), b1=stats.fit(feature_indices=(0, 1, 2, 3, 4)))
        for row in outputs:
            for name in ("ordered_top5", "ordered_top10", "ordered_top25"):
                self.assertEqual(len(row[name]), len(set(row[name])))
                self.assertLessEqual(len(row[name]), 15)

    def test_full_prefix_and_centered_neighbor_predictions_match(self):
        batch = measure_features(self.views, self.queries, (4, 5), self.context, retain_comparators=True)
        stats = RidgeMoments()
        stats.add(batch.features, response_matrix(self.queries, batch, self.context), batch.allowed, batch.fit_eligible)
        outputs, _ = query_predictions(batch, self.context, self.views,
            b0=stats.fit(feature_indices=(0, 1, 2, 3)), b1=stats.fit(feature_indices=(0, 1, 2, 3, 4)))
        for row in outputs:
            self.assertEqual(row["centered_nearest"], row["prefix_nn_003"])

    def test_rank_ties_and_nonfinite_exclusion(self):
        ranking = rank_scores(np.array([1., 1., np.nan, 1.]), np.array([True, True, True, False]),
                              np.array([2, 3, 9, 9]))
        self.assertEqual(ranking, (1, 0))


class FoldAndRidgeTests(unittest.TestCase):
    def test_stratified_group_folds_are_balanced_and_order_invariant(self):
        queries = [Query(str(i), f"word_{i}", "Concept" if i < 11 else "Entity", ("label",))
                   for i in range(19)]
        queries.append(replace(queries[0], identifier="repeat"))
        first = lexical_folds(queries, range(20), folds=5, seed="fixture")
        second = lexical_folds(queries, reversed(range(20)), folds=5, seed="fixture")
        self.assertEqual(first, second)
        self.assertEqual(first[0], first[19])
        for category in ("Concept", "Entity"):
            counts = [sum(q.category == category and first[i] == fold for i, q in enumerate(queries[:19]))
                      for fold in range(5)]
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_conflicting_lexical_group_strata_rejected(self):
        queries = [Query("a", "word", "Concept", ("label",)),
                   Query("b", "word", "Entity", ("label",))]
        with self.assertRaises(ValueError):
            lexical_folds(queries, (0, 1), folds=2, seed="fixture")

    def test_query_balanced_ridge_matches_direct_weighted_lstsq(self):
        rng = np.random.default_rng(149)
        x = rng.normal(size=(4, 5, 3))
        y = rng.integers(0, 2, size=(4, 5)).astype(float)
        allowed = np.array([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1],
                            [1, 0, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=bool)
        eligible = np.array([True, True, True, False])
        stats = RidgeMoments(features=3)
        stats.add(x, y, allowed, eligible)
        fitted = stats.fit(feature_indices=(0, 1, 2), regularization=.1)
        values, targets, weights = [], [], []
        for row in range(3):
            for col in np.flatnonzero(allowed[row]):
                values.append(x[row, col]); targets.append(y[row, col])
                weights.append(1 / (3 * allowed[row].sum()))
        values, targets, weights = np.asarray(values), np.asarray(targets), np.asarray(weights)
        mean = (weights[:, None] * values).sum(axis=0)
        scale = np.sqrt((weights[:, None] * (values - mean) ** 2).sum(axis=0))
        design = np.column_stack((np.ones(len(values)), (values - mean) / scale))
        penalty = np.column_stack((np.zeros(3), np.sqrt(.1) * np.eye(3)))
        direct, *_ = np.linalg.lstsq(np.vstack((np.sqrt(weights[:, None]) * design, penalty)),
                                    np.r_[np.sqrt(weights) * targets, np.zeros(3)], rcond=None)
        np.testing.assert_allclose(fitted.mean, mean, rtol=0, atol=1e-14)
        np.testing.assert_allclose(fitted.scale, scale, rtol=0, atol=1e-14)
        np.testing.assert_allclose(np.r_[fitted.intercept, fitted.coefficient], direct, rtol=0, atol=1e-13)
        self.assertEqual(stats.queries, 3)
        self.assertEqual(stats.candidate_examples, 10)

    def test_matched_models_share_base_scaling_exactly(self):
        rng = np.random.default_rng(53)
        features = rng.normal(size=(3, 7, 5))
        features[:, :, 3] = 1
        response = rng.integers(0, 2, size=(3, 7)).astype(float)
        stats = RidgeMoments()
        stats.add(features, response, np.ones((3, 7), bool), np.ones(3, bool))
        b0 = stats.fit(feature_indices=(0, 1, 2, 3))
        b1 = stats.fit(feature_indices=(0, 1, 2, 3, 4))
        np.testing.assert_array_equal(b0.mean, b1.mean[:4])
        np.testing.assert_array_equal(b0.scale, b1.scale[:4])
        self.assertEqual(b0.scale[3], 1)
        self.assertTrue(np.isfinite(b1.predict(features)).all())

    def test_invalid_numerical_inputs_and_no_fit_queries(self):
        with self.assertRaises(ValueError):
            prepare_views([[1., np.nan], [2., 1.]], np.ones(2, bool),
                          mean=np.zeros(2), components=np.eye(2), dimensions=(1, 2))
        with self.assertRaises(ValueError):
            prepare_views([[1., 2.], [2., 1.]], np.ones(2, bool),
                          mean=np.zeros(2), components=np.ones((2, 2)), dimensions=(1, 2))
        with self.assertRaises(ValueError):
            RidgeMoments().fit(feature_indices=(0, 1, 2, 3))


if __name__ == "__main__":
    unittest.main()
