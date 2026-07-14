# tulip leaderboard

A committed, **byte-for-byte regenerable** leaderboard for Polish dialect
identification. There is no widely adopted benchmark for this task, so
comparable numbers require everyone to train on *identical, frozen,
speaker-disjoint splits*. This directory is that shared reference point.

Everything here runs on the offline **`synthetic`** corpus, so the whole suite
regenerates with no downloads and no locally acquired data.

> **What these numbers are not.** `synthetic` is a generated benchmark fixture,
> not real speech. Scores here measure whether a model can pick up injected
> lexical markers and phonological transforms — they say **nothing** about
> real-world dialect identification accuracy. The suite exists to make the
> machinery comparable and reproducible, not to claim a result.

## Layout

| Path | Purpose |
| :--- | :--- |
| `suite.yaml` | The `LeaderboardSuite`: which experiment configs to run, and which competitor models to apply to each. |
| `configs/*.yaml` | Runnable `ExperimentConfig`s (feature "tracks"), one benchmark run each. |
| `results/<suite>/` | The committed artifacts (see below). |

The competitor set is classical only (`naive_bayes`, `logistic_regression`,
`linear_svm`, `random_forest`) so the suite needs no heavy or optional
dependencies.

## Regenerate

Run from the **repository root** (relative config paths in `suite.yaml` resolve
against the working directory):

```bash
tulip leaderboard benchmarks/suite.yaml --out benchmarks/results
```

This trains every `(config, model)` pair on its frozen split and writes into
`benchmarks/results/<suite-name>/`:

| File | Contents | Deterministic? | Committed? |
| :--- | :--- | :--- | :--- |
| `leaderboard.md` | The ranking table (sorted by macro-F1), with per-model **ECE** and **Brier** calibration columns. | **Yes** | Yes |
| `provenance.json` | Audit record: tulip version, per-config seeds, split sizes and class distribution, fixed-precision per-result metrics (incl. ECE/Brier), and an `environment` block (Python floor, key dependency versions from `uv.lock`, and content digests of the configs and lexicons). | **Yes** | Yes |
| `significance-<experiment>.{md,json}` | Per-experiment paired significance: bootstrap CIs per metric, exact Holm-corrected McNemar between every model pair, and a "tied with best" grouping. | **Yes** | Yes |
| `leaderboard.json` | The full raw benchmark dump (round-trips via `load_benchmark`). | No — carries wall-clock timings. | No (gitignored) |

The **ECE/Brier** columns come from the suite's `calibration_bins` (10 uniform
bins): two models can share an accuracy yet differ sharply in how honest their
confidence is. The **significance** files are what let you read the ranking
correctly — on a small split, the top few models are often statistically tied.

Trained models, frozen splits, and per-split reports go to each config's
`output_dir` (`artifacts/benchmarks/`, gitignored) rather than into this
directory, so only the deterministic artifacts are ever committed.

## The reproducibility guarantee

For a **fixed seed and environment**, re-running the suite produces
**byte-identical** `leaderboard.md`, `provenance.json`, and the
`significance-*` artifacts. Verify with the built-in gate, which regenerates the
suite into a scratch directory and byte-compares every guaranteed artifact
against the committed ones (exit non-zero on any drift):

```bash
tulip repro verify benchmarks/suite.yaml
```

CI runs the same suite twice and diffs the outputs, so a hidden nondeterminism —
an unseeded RNG, a dict-ordering leak, a moved generator default — fails the
build. (CI checks same-run determinism rather than a match against the committed
board, because BLAS differences across operating systems can perturb the last
ULP of a probability-derived metric; `tulip repro verify` performs the stricter
match on the platform that produced the board.)

It holds by construction:

- **Frozen splits.** Every competitor trains on the same speaker-disjoint split,
  built deterministically from each config's `seed` / `split.seed`.
- **Pinned task difficulty.** Each config pins the generator's `marker_dropout`,
  `noise_level`, and `seed` rather than inheriting library defaults, so a
  changed default cannot silently move the numbers.
- **No wall-clock time in the artifacts.** `leaderboard.md` renders only
  `{experiment, model, accuracy, f1_macro, f1_weighted, roc_auc, n_train}` —
  never the nondeterministic `wall_seconds` the raw benchmark table carries.
- **Total, stable ordering.** Rows sort by descending macro-F1, ties broken by
  `(experiment, model)`, so reordering the inputs never changes the table.
- **Deterministic serialisation.** `provenance.json` and the significance
  artifacts are written with sorted keys, no timestamps, no timings, and every
  float at a fixed precision.
- **Environment pinned from the lockfile.** The `environment` block reads its
  dependency versions from the committed `uv.lock` (not the live interpreter),
  so it is identical across machines; a bump to numpy/scipy/scikit-learn/pandas
  changes it — and the gate flags it, because those versions can move a metric.

`leaderboard.json` is intentionally excluded from the guarantee — it preserves
raw per-model timings for reference, and is gitignored for exactly that reason.

## Reading the table

Rows are identified by `(experiment, model)`, not by model alone: a suite trains
every competitor against every config, so the same model name appears once per
track.

The two shipped tracks are a deliberate contrast. `char_baseline` uses character
n-grams, which can see the phonological transforms (*mazurzenie*,
soft-labials); `lexical_baseline` uses word-level TF-IDF and stylometry, which
can only see whole-word markers. On the samples where `marker_dropout` removed
those markers, the lexical track has strictly less to go on — and the leaderboard
shows it.

## Rank a single track

To benchmark one config's competitors interactively (Rich table, no committed
artifact):

```bash
tulip benchmark benchmarks/configs/char_baseline.yaml -m naive_bayes -m logistic_regression
```

## Add a competitor or a track

- **New competitor model:** add its registry name to `suite.yaml`'s `models`.
- **New feature track:** add a `configs/<name>.yaml` (mirror an existing one;
  keep `data.datasets` on `synthetic` and pin its `params`) and list it under
  `suite.yaml`'s `configs`.

Then regenerate and commit the updated `leaderboard.md`, `provenance.json`, and
`significance-*` files. Run `tulip repro verify benchmarks/suite.yaml` first to
confirm the board still reproduces.
