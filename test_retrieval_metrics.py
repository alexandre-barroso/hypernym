import unittest
from retrieval_metrics import evaluate


class MetricTests(unittest.TestCase):
    def test_first_and_second_relevant_are_distinct_formulas(self):
        score=evaluate(['wrong','target'],['target'])
        self.assertEqual(score.reciprocal_rank,.5)
        self.assertEqual(score.ap_capped_gold,.5)
        self.assertAlmostEqual(score.semeval_mean_normalized_prefix_precision,14/15)
        self.assertEqual(score.precision,1/15)
        self.assertEqual(score.recall,1)

    def test_all_missing(self):
        score=evaluate([],['one','two'])
        self.assertEqual(score.relevance,(0,)*15)
        self.assertEqual(score.ap_capped_gold,0)
        self.assertEqual(score.semeval_mean_normalized_prefix_precision,0)

    def test_limit_before_deduplication(self):
        score=evaluate(['A']*15+['b'],['a','b'])
        self.assertEqual(score.predictions,('a',))
        self.assertEqual(score.precision,1/15)
        self.assertEqual(score.recall,.5)
        self.assertEqual(score.ap_capped_gold,.5)

    def test_gold_denominator_is_explicit(self):
        gold=[f'w{i}' for i in range(20)]
        score=evaluate(gold[:15],gold)
        self.assertEqual(score.ap_capped_gold,1)
        self.assertEqual(score.ap_full_gold,.75)
        self.assertEqual(score.recall,.75)
        self.assertEqual(score.semeval_mean_normalized_prefix_precision,1)

    def test_invalid_labels(self):
        for labels in ([],['a','a'],['A'],['']):
            with self.assertRaises(ValueError):evaluate([],labels)
        with self.assertRaises(ValueError):evaluate([''],['a'])
        with self.assertRaises(ValueError):evaluate([],['a'],0)


if __name__=='__main__':unittest.main()
