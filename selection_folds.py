"""Original deterministic training-fold construction, extracted unchanged."""

from label_transfer import lexical_folds

OUTER_SEED = "label-transfer-r3-20260926"

def assert_group_disjoint(queries, left, right):
    overlap = {queries[i].word for i in left} & {queries[i].word for i in right}
    if overlap:
        raise ValueError("Lexical-type overlap across fold roles")

def fold_manifest(queries):
    all_rows = tuple(range(len(queries)))
    assignment = lexical_folds(queries, all_rows, folds=5, seed=OUTER_SEED)
    folds = []
    for fold in range(5):
        validation = [i for i in all_rows if assignment[i] == fold]
        training = [i for i in all_rows if assignment[i] != fold]
        assert_group_disjoint(queries, training, validation)
        inner_seed = OUTER_SEED + f"|outer={fold}|inner"
        inner = lexical_folds(queries, training, folds=3, seed=inner_seed)
        inner_records = []
        for inner_fold in range(3):
            feature_queries = [i for i in training if inner[i] == inner_fold]
            references = [i for i in training if inner[i] != inner_fold]
            assert_group_disjoint(queries, feature_queries, references)
            assert_group_disjoint(queries, validation, references + feature_queries)
            inner_records.append({"inner_fold": inner_fold, "feature_query_rows": feature_queries,
                                  "reference_rows": references})
        folds.append({"outer_fold": fold, "validation_rows": validation,
                      "training_rows": training, "inner_seed": inner_seed, "inner_folds": inner_records})
    return {"outer_seed": OUTER_SEED, "outer_folds": 5, "inner_folds": 3,
            "queries": [{"row": i, "id": q.identifier, "word": q.word,
                         "category": q.category, "fold": assignment[i]} for i, q in enumerate(queries)],
            "folds": folds}
