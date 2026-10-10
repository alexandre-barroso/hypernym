# Dimensional recurrence in hypernym label transfer

Original numerical implementation and portable reproduction for a matched comparison of full-space lexical transfer with dimensional recurrence. The target is agreement with supplied hypernym references. A geometric recurrence fraction is not a probability of linguistic correctness. Citation file hidden during peer review.

## Environment and checks

Use Python 3.12.3 and NumPy 2.4.4 in a local virtual environment.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -B verify.py
```

The verification command has nonzero failure exits. Without an evaluated run it checks the fifty synthetic and numerical tests and explicitly reports that corpus reproduction has not been performed.

## Public inputs

Obtain the three archives identified by URL and SHA-256 in `inputs.json`: SemEval-2018 Task 9, GloVe Wiki/Giga 2024 50-dimensional vectors and the released fastText WikiNews subword 300-dimensional text vectors. Downloads are separate and subject to their upstream terms. No datasets, copyrighted reference full texts or original unpublished source files are bundled. The fastText header and actual row count are 999,994; the nominal “1M” filename is not an exact row count.

The following example assumes those files are in `inputs/`. All archive hashes are verified before reading, and only declared ZIP members are read; archives are not extracted to arbitrary paths.

```sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
.venv/bin/python -B pipeline.py vectors --resource glove --run run --dataset-zip inputs/SemEval2018-Task9.zip --vectors-zip inputs/glove.2024.wikigiga.50d.zip
.venv/bin/python -B pipeline.py vectors --resource fasttext --run run --dataset-zip inputs/SemEval2018-Task9.zip --vectors-zip inputs/wiki-news-300d-1M-subword.vec.zip
.venv/bin/python -B pipeline.py fit --resource glove --run run --dataset-zip inputs/SemEval2018-Task9.zip
.venv/bin/python -B pipeline.py fit --resource fasttext --run run --dataset-zip inputs/SemEval2018-Task9.zip
.venv/bin/python -B pipeline.py predict --resource glove --run run --dataset-zip inputs/SemEval2018-Task9.zip
.venv/bin/python -B pipeline.py predict --resource fasttext --run run --dataset-zip inputs/SemEval2018-Task9.zip
.venv/bin/python -B pipeline.py score --run run --dataset-zip inputs/SemEval2018-Task9.zip
.venv/bin/python -B verify.py --results run
```

Use a fresh output directory; existing models/predictions are preserved. The full run scores all 418 distinct systems and recomputes 100,000 paired bootstrap draws with the original category-membership grouping. Prediction performs one warm-up and five measured repetitions in shuffled order. Wall time depends on hardware and load; reproducing the scores does not reproduce the original seconds. The two resources should be timed sequentially on an otherwise idle machine for comparison with the reported timing scope.

For a smaller integration check, add `--selected-only --repeats 1` to **both** prediction commands. This checks the three selected families per resource. The scorer then compares every mean metric for those six systems, while explicitly skipping the full contrast/bootstrap report. This shorter mode must not be described as a replay of all 418 systems or all confidence intervals.

Training-only selection can also be replayed after the two `vectors` commands. It uses the same five outer, three middle and three inner folds, compares all 21 distinct configurations, and reconstructs the complete 49- and 299-prefix selection curves. It reads training references only; it does not open test references.

```sh
.venv/bin/python -B training_selection.py --resource glove --run run --dataset-zip inputs/SemEval2018-Task9.zip --output selection
.venv/bin/python -B training_selection.py --resource fasttext --run run --dataset-zip inputs/SemEval2018-Task9.zip --output selection
.venv/bin/python -B verify.py --selection selection
```

Each resource writes its own subdirectory. The replay verifies the original family choices, penalties and selected single-prefix dimensions against the stored training outcomes with absolute tolerance `1e-12`. This is a numerical replay of the documented selection, not a new prospective experiment.

## Methods and result map

- `label_transfer*.py`: reference contexts, deterministic ranking, query-weighted ridge sufficient statistics, cross-fitting roles and feature definitions.
- `predict_label_transfer_final.py`: complete resident-input predictors, without test labels or fitting.
- `score_label_transfer_final.py`, `retrieval_metrics.py`, `query_structure_amendment.py`: fixed-depth metrics, candidate ceilings, contrasts and paired resampling.
- `pipeline.py`: portable archive authentication, streaming covariance/PCA, fixed-configuration refitting, prediction and scoring.
- `training_selection.py`, `selection_folds.py`: optional complete replay of training selection and single-prefix curves.
- `configurations.json`: original training-only selections, penalty grids, schedules and controls. Reproducing them does not imply that selection was repeated in the portable fixed-configuration run.
- `*_expected_models.json`: original fitted coefficients, scaling parameters and direct weighted-fit checks.
- `expected_results.json`: original aggregate results for all systems and primary interval targets.

The pipeline recreates all nineteen final fits per resource from training-only cross-fitted features, comparing fitted parameters to the original with `rtol=1e-8, atol=1e-10`. It excludes training-overlap query rows 185, 281 and 1390 before prediction and retains both category rows of the mixed-category lexical type. The score stage alone opens test references. This reproduction is not a new preregistration or an erasure of earlier development exposure.

The paper reports small, heterogeneous conditional increments. Both primary adjusted intervals include zero; this is neither superiority nor equivalence. Candidate limits, missing-vector fallback and incomplete references remain part of the evaluation. Derived counts and model parameters can be checked locally, but numerical verification does not establish novelty, linguistic truth or practical importance in another application.

The editable scientific figure is regenerated by `python plot_figure.py` after installing `requirements-figures.txt`. It writes PDF, PNG (450 dpi), JPEG and the plotted numeric data to `figures/`. Optional `MANUSCRIPT_AUTHOR` sets authorship metadata for a named author copy; the default is neutral.
