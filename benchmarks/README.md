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
| `leaderboard.md` | The ranking table (sorted by macro-F1). | **Yes** | Yes |
| `provenance.json` | Audit record: tulip version, per-config seeds, split sizes and class distribution, fixed-precision per-result metrics. | **Yes** | Yes |
| `leaderboard.json` | The full raw benchmark dump (round-trips via `load_benchmark`). | No — carries wall-clock timings. | No (gitignored) |

Trained models, frozen splits, and per-split reports go to each config's
`output_dir` (`artifacts/benchmarks/`, gitignored) rather than into this
directory, so only the deterministic artifacts are ever committed.

## The reproducibility guarantee

For a **fixed seed and environment**, re-running the suite produces
**byte-identical `leaderboard.md` and `provenance.json`**. Verify with:

```bash
tulip leaderboard benchmarks/suite.yaml --out /tmp/lb
diff benchmarks/results/synthetic-leaderboard/leaderboard.md /tmp/lb/synthetic-leaderboard/leaderboard.md
```

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
- **Deterministic serialisation.** `provenance.json` is written with sorted
  keys, no timestamps, no timings, and every float at a fixed precision.

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

Then regenerate and commit the updated `leaderboard.md` and `provenance.json`.
