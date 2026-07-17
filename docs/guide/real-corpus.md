# Running the real dialect benchmark

The committed leaderboard runs on a synthetic fixture, so its numbers are not real
dialect accuracy. To produce **real** numbers you assemble the dialektarium and
dgp corpora locally (a licence-bound, manual step tulip cannot automate), then run
the benchmark from your own machine. The raw data never leaves it; only digests
and aggregate metrics are committed, so the benchmark stays reproducible in
principle without redistributing the corpus.

## 1. The manifest schema

Each corpus lives under `data/raw/<name>/` as a `manifest.{csv,tsv,jsonl}` (plus a
`clips/` directory for audio). The canonical columns are:

| Column | Meaning |
| :--- | :--- |
| `id` | Unique sample id. |
| `text` | The transcribed dialect text (one of `text`/`audio_path` is required). |
| `audio_path` | Clip path relative to the manifest, for audio corpora. |
| `speaker_id` | Speaker identifier; drives speaker-disjoint splitting. When absent, a surrogate id is synthesised from the geography columns. |
| `family`, `dialect`, `region`, `village`, `voivodeship` | The label at each taxonomy level; a corpus fills whichever levels it annotated. |

Any other columns are preserved as free-form metadata. In particular **`age` and
`gender`**, if present, are read by the demographic-bias slicing (`age` is bucketed
into bands; already-banded strings such as `thirties` pass through).

Confirm the manifest before building:

```
tulip data validate data/raw/dialektarium/manifest.csv
```

## 2. Build, benchmark, document, verify

The steps below compose the already-tested commands. Run them per corpus for the
datasheets, and once for the combined board.

```
# (a) Run the real board (majority floor + the four classical models).
tulip leaderboard benchmarks/real_text_suite.yaml --out artifacts/real-text

# (b) A datasheet per source corpus (fill the prose spec first).
tulip data prepare benchmarks/configs/real_text_char.yaml --out artifacts/build/dialektarium
tulip card datasheet artifacts/build/dialektarium \
    --spec benchmarks/datasheets/dialektarium.yaml --dataset dialektarium \
    --out artifacts/real-text/datasheet-dialektarium.md

# (c) The subgroup bias analysis (stays local; paste its summary into the report).
tulip analyze artifacts/real-text/predictions_test.json --fairness \
    --out artifacts/real-text/bias.md

# (d) The paper-style methodology-and-results report.
tulip card benchmark artifacts/real-text \
    --datasheet artifacts/real-text/datasheet-dialektarium.md \
    --bias artifacts/real-text/bias.md --out docs/benchmark.md

# (e) The reproducibility check: rebuild the split from your local data and
#     confirm it matches the committed fingerprint.
tulip repro verify-lock benchmarks/configs/real_text_char.yaml \
    artifacts/real-text/split_lock.json
```

## 3. What to commit

Commit only the digests and aggregates under
`benchmarks/results/real-text-leaderboard/` (`split_lock.json`, `provenance.json`,
`leaderboard.md`, `significance-*`), plus the assembled `docs/benchmark.md` and the
filled-in datasheets. Never commit the corpus text, audio, or the per-sample
predictions. See `benchmarks/results/real-text-leaderboard/README.md` for the
exact policy.

## 4. A note on split viability

Speaker-disjoint splitting needs enough distinct speaker groups per class; with
small corpora and surrogate speaker ids, a 70/15/15 split across several classes
can be tight. `tulip data prepare` fails loudly if a class cannot be split, so run
it first and widen the corpus or coarsen the target level (for example benchmark
at `family` rather than `dialect`) if it does.
