"""Explicit shortlist metrics and a named SemEval-2018 compatibility score.

These metrics measure agreement with supplied incomplete reference labels.
They do not establish that an unlisted lexical answer is linguistically false.
No organizer code is imported or executed. Formulas were independently written
after inspecting the public updated scorer.
"""
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class RetrievalMetrics:
    predictions: tuple[str,...]
    relevance: tuple[int,...]
    gold_count: int
    depth: int
    reciprocal_rank: float
    precision: float
    recall: float
    ap_capped_gold: float
    ap_full_gold: float
    semeval_mean_normalized_prefix_precision: float
    semeval_normalized_precision: float


def evaluate(predictions: Sequence[str], gold: Sequence[str], depth: int=15) -> RetrievalMetrics:
    """Evaluate at a fixed output depth, padding unreturned positions with misses.

    Like the updated organizer script, limit before deduplicating predictions;
    use str.lower(), not casefold(). Require unique lower-case nonempty gold.
    AP_capped divides precision-at-relevant-rank sum by min(depth, gold_count).
    AP_full divides by the complete reference-list size. Both are explicit;
    neither is silently interchanged with the organizer's different 'MAP'.

    Organizer compatibility score = mean_{k=1..depth} hits(k)/min(k,G).
    It averages ALL prefix ranks, including ranks with an irrelevant result.
    This is reported by its formula, not as conventional average precision.
    """
    if isinstance(depth,bool) or not isinstance(depth,int) or depth<1:
        raise ValueError('depth must be a positive integer')
    labels=tuple(gold)
    if not labels or any(not isinstance(x,str) or not x or x!=x.lower() for x in labels):
        raise ValueError('gold must contain nonempty lowercase labels')
    if len(set(labels))!=len(labels):raise ValueError('duplicate gold labels')
    supplied=tuple(predictions)
    if any(not isinstance(x,str) or not x for x in supplied):raise ValueError('invalid prediction')
    unique=[];seen=set()
    for term in supplied[:depth]:
        term=term.lower()
        if term not in seen:unique.append(term);seen.add(term)
    relevant=tuple(int(x in labels) for x in unique)+(0,)*(depth-len(unique))
    hits=0;first=0.;ap_sum=0.;compat_sum=0.
    for rank,hit in enumerate(relevant,1):
        hits+=hit
        if hit:
            if first==0:first=1/rank
            ap_sum+=hits/rank
        compat_sum+=hits/min(rank,len(labels))
    return RetrievalMetrics(tuple(unique),relevant,len(labels),depth,first,
        hits/depth,hits/len(labels),ap_sum/min(depth,len(labels)),ap_sum/len(labels),
        compat_sum/depth,hits/min(depth,len(labels)))
