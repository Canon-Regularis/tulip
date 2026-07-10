# Synthetic corpora

Every real corpus tulip supports needs a licence and a download. The `synthetic`
corpus needs neither: it is **generated in-process**, so a fresh clone can run
the whole toolkit — generate, split, train, evaluate, benchmark, serve — end to
end with no acquisition.

!!! danger "A benchmark fixture, not real speech or text"
    Scores on the synthetic corpus say **nothing** about real-world Polish
    dialect identification. It exists to make the machinery runnable and its
    outputs reproducible, not to estimate accuracy. Never quote a synthetic
    score as a dialect-ID result.

## What it is for

The synthetic corpus provides a *reproducible, learnable* task grounded in
tulip's own linguistic resources, so that:

- the pipeline can be exercised end to end on any machine, offline;
- CI and examples produce byte-identical artifacts for a fixed seed;
- speaker-disjoint splitting is genuinely tested rather than trivially satisfied.

## What signal it contains

The generator layers three kinds of signal, each mirroring something a real
model must cope with:

- **Lexical** — genuine marker lexemes drawn from the project's own lexicon
  (`src/tulip/features/text/lexicons/dialect_markers.yaml`): Podhale *baca*,
  Silesian *gryfny*, Kashubian *chëcz*, and so on.
- **Phonological** — real sound changes applied as deterministic string
  transforms: *mazurzenie* (cz/sz/ż → c/s/z) for the Masovian group, and
  asynchronous soft-labials (pi → psi, bi → bzi) for Kurpie. These are exactly
  what character n-grams can see and a whole-word lexicon cannot.
- **Speaker idiolect** — each speaker gets a personal filler vocabulary, so a
  model can partly re-identify the speaker. That is precisely the leakage that
  speaker-disjoint splitting must defend against.

## The knob that matters: `marker_dropout`

`marker_dropout` (default `0.20`) is the fraction of samples carrying **no**
lexical marker at all, mirroring the fact that plenty of real dialect utterances
contain no diagnostic lexeme. It gives the task an irreducible error floor. Set
it to `0.0` and every linear model scores a perfect `1.000` — a benchmark that
cannot rank anything.

Generator parameters flow through `params:` like any other loader:

```yaml
data:
  datasets:
    - name: synthetic
      params:
        n_speakers_per_dialect: 12
        samples_per_speaker: 12
        include_standard: false
        marker_dropout: 0.20
        seed: 7
```

## Running it

```bash
tulip train configs/synthetic_text.yaml     # dialect level
tulip train configs/synthetic_family.yaml   # family level, incl. the `standard` class
```

To materialise an auditable copy on disk (a JSONL manifest, byte-identical for a
given seed):

```bash
tulip data synthesize --out data/raw/synthetic --seed 7
```

## Audio parity

The synthetic path is not text-only: an audio fixture mirrors the same generate →
split → train → evaluate flow so audio pipelines are exercised offline as well.
Its scores carry the same caveat — a fixture, not real speech.

## Under the hood

The generator is part of the public data API: `generate_corpus` and
`SyntheticSpec` produce the samples, and `write_synthetic_manifest` persists
them. See the [data reference](../reference/data.md).

## Learn more

For the full dataset story — acquisition of the *real* corpora, the manifest
format, speaker-ID surrogates, and the reproducible-split machinery — see
[Datasets](../datasets.md).
