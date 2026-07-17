# Real-text leaderboard results

This directory holds the committed artifacts of the **real** speaker-disjoint
text benchmark (dialektarium + dgp), produced by
`benchmarks/real_text_suite.yaml`. It is the counterpart to the
`synthetic-leaderboard/` fixture: those numbers are synthetic and prove nothing
about real accuracy; the numbers here are real.

## What lives here (digests and aggregates only)

The raw dialektarium and dgp corpora are assembled manually and are **not
redistributable**, so the data itself is never committed. What is committed makes
the benchmark reproducible *in principle* without the data:

- `split_lock.json` (one per config): the per-split BLAKE2b content fingerprint of
  the frozen speaker-disjoint split. Anyone with the same local corpus can rebuild
  the split and confirm it matches, byte for byte, with
  `tulip repro verify-lock <config.yaml> <split_lock.json>`.
- `provenance.json`: the environment (locked library versions), per-config seeds
  and split sizes, class distributions, and the dataset digest, all sorted and
  timestamp-free.
- `leaderboard.md`: the ranked results table (accuracy, macro/weighted F1, ROC
  AUC, calibration), floored by the `majority` baseline.
- `significance-*.md` / `significance-*.json`: the paired bootstrap confidence
  intervals and exact McNemar tests against the floor.

## What never lives here

The corpus text, audio, per-sample predictions, or anything from which the raw
data could be reconstructed. Per-sample fairness/bias analysis is run locally with
`tulip analyze <predictions> --fairness` and its summary can be pasted into the
benchmark report, but the predictions themselves stay private.

## Producing and verifying it

```
tulip benchmark real benchmarks/real_text_suite.yaml --data-root data/raw --out artifacts/real-text
# review artifacts/real-text, then commit the four artifact kinds above here
tulip repro verify-lock benchmarks/configs/real_text_char.yaml split_lock.json
```

See `docs/guide/real-corpus.md` for the full runbook and the canonical manifest
schema, and `docs/benchmark.md` for the assembled methodology-and-results report.
