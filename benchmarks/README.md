# tulip leaderboard

A committed, byte-for-byte regenerable leaderboard for Polish dialect
identification. There is no widely adopted benchmark for this task. Comparable
numbers require everyone to train on identical, frozen, speaker-disjoint splits.
This directory is that shared reference point.

Everything here runs on the offline `synthetic` corpus. The whole suite
regenerates with no downloads and no locally acquired data.

> **What these numbers are not.** `synthetic` is a generated fixture, not real
> speech. The scores measure whether a model can pick up injected lexical markers
> and phonological transforms. They say nothing about real dialect accuracy. The
> suite exists to make the machinery comparable and reproducible, not to claim a
> result.

## Layout

| Path | Purpose |
| :--- | :--- |
| `suite.yaml` | The `LeaderboardSuite`: which configs to run, and which models to apply to each. |
| `configs/*.yaml` | Runnable `ExperimentConfig`s (feature "tracks"), one benchmark run each. |
| `results/<suite>/` | The committed artifacts (see below). |

The competitor set is classical only (`naive_bayes`, `logistic_regression`,
`linear_svm`, `random_forest`). The suite needs no heavy or optional
dependencies.

## Regenerate

Run from the repository root. The relative config paths in `suite.yaml` resolve
against the working directory.

```bash
tulip leaderboard benchmarks/suite.yaml --out benchmarks/results
```

This trains every `(config, model)` pair on its frozen split. It writes into
`benchmarks/results/<suite-name>/`:

| File | Contents | Deterministic? | Committed? |
| :--- | :--- | :--- | :--- |
| `leaderboard.md` | The ranking table (sorted by macro-F1), with per-model ECE and Brier calibration columns. | Yes | Yes |
| `provenance.json` | Audit record: tulip version, per-config seeds, split sizes and class distribution, fixed-precision per-result metrics (incl. ECE/Brier), and an `environment` block. | Yes | Yes |
| `significance-<experiment>.{md,json}` | Per-experiment paired significance: bootstrap CIs, Holm-corrected McNemar between every model pair, and a "tied with best" grouping. | Yes | Yes |
| `leaderboard.json` | The full raw benchmark dump (round-trips via `load_benchmark`). | No. Carries wall-clock timings. | No (gitignored) |

The ECE and Brier columns come from the suite's `calibration_bins` (10 uniform
bins). Two models can share an accuracy yet differ sharply in how honest their
confidence is. The significance files let you read the ranking correctly. On a
small split, the top few models are often statistically tied.

Trained models, frozen splits, and per-split reports go to each config's
`output_dir` (`artifacts/benchmarks/`, gitignored). Only the deterministic
artifacts are committed.

## The reproducibility guarantee

For a fixed seed and environment, re-running the suite produces byte-identical
`leaderboard.md`, `provenance.json`, and `significance-*` artifacts. Verify with
the built-in gate. It regenerates the suite into a scratch directory and
byte-compares every guaranteed artifact against the committed ones. It exits
non-zero on any drift.

```bash
tulip repro verify benchmarks/suite.yaml
```

CI runs the same suite twice and diffs the outputs. A hidden nondeterminism (an
unseeded RNG, a dict-ordering leak, a moved generator default) fails the build.
CI checks same-run determinism, not a match against the committed board. BLAS
differences across operating systems can perturb the last ULP of a
probability-derived metric. `tulip repro verify` performs the stricter match on
the platform that produced the board.

The guarantee holds by construction:

- **Frozen splits.** Every competitor trains on the same speaker-disjoint split,
  built deterministically from each config's `seed` and `split.seed`.
- **Pinned task difficulty.** Each config pins the generator's `marker_dropout`,
  `noise_level`, and `seed`. It does not inherit library defaults, so a changed
  default cannot silently move the numbers.
- **No wall-clock time in the artifacts.** `leaderboard.md` renders only
  `{experiment, model, accuracy, f1_macro, f1_weighted, roc_auc, n_train}`. It
  never renders the nondeterministic `wall_seconds`.
- **Total, stable ordering.** Rows sort by descending macro-F1, ties broken by
  `(experiment, model)`. Reordering the inputs never changes the table.
- **Deterministic serialisation.** `provenance.json` and the significance
  artifacts use sorted keys, no timestamps, no timings, and a fixed float
  precision.
- **Environment pinned from the lockfile.** The `environment` block reads its
  dependency versions from the committed `uv.lock`, not the live interpreter. It
  is identical across machines. A bump to numpy, scipy, scikit-learn, or pandas
  changes it, and the gate flags it, because those versions can move a metric.

`leaderboard.json` is excluded from the guarantee. It keeps raw per-model timings
for reference, so it is gitignored.

## Reading the table

Rows are identified by `(experiment, model)`, not by model alone. A suite trains
every competitor against every config, so the same model name appears once per
track.

The two shipped tracks are a contrast. `char_baseline` uses character n-grams. It
can see the phonological transforms (*mazurzenie*, soft labials).
`lexical_baseline` uses word-level TF-IDF and stylometry. It can see only
whole-word markers. On samples where `marker_dropout` removed those markers, the
lexical track has less to go on, and the leaderboard shows it.

## Rank a single track

To benchmark one config's competitors interactively (Rich table, no committed
artifact):

```bash
tulip benchmark benchmarks/configs/char_baseline.yaml -m naive_bayes -m logistic_regression
```

## Add a competitor or a track

- **New competitor model.** Add its registry name to `suite.yaml`'s `models`.
- **New feature track.** Add a `configs/<name>.yaml` (mirror an existing one, keep
  `data.datasets` on `synthetic`, and pin its `params`). List it under
  `suite.yaml`'s `configs`.

Then regenerate and commit the updated `leaderboard.md`, `provenance.json`, and
`significance-*` files. Run `tulip repro verify benchmarks/suite.yaml` first to
confirm the board still reproduces.
