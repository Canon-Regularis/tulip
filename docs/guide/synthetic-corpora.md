# Synthetic corpora

Every real corpus tulip supports needs a licence and a download. The `synthetic`
corpus needs neither. It is generated in memory. A fresh clone can run the whole
toolkit offline: generate, split, train, evaluate, benchmark, serve.

!!! danger "A benchmark fixture, not real speech or text"
    Scores on the synthetic corpus say nothing about real Polish dialect
    identification. The corpus exists to make the machinery runnable and its
    outputs reproducible. Never quote a synthetic score as a dialect-ID result.

## What it is for

The synthetic corpus gives you a reproducible, learnable task. It is grounded in
tulip's own linguistic resources. It lets you:

- run the pipeline end to end on any machine, offline;
- produce byte-identical artifacts for a fixed seed, in CI and examples;
- exercise speaker-disjoint splitting for real, not trivially.

## What signal it contains

The generator layers three kinds of signal. Each mirrors something a real model
must handle.

- **Lexical.** Real marker lexemes from the project lexicon
  (`src/tulip/features/text/lexicons/dialect_markers.yaml`). For example: Podhale
  *baca*, Silesian *gryfny*, Kashubian *chëcz*.
- **Phonological.** Real sound changes applied as deterministic string
  transforms. *Mazurzenie* (cz/sz/ż to c/s/z) for the Masovian group.
  Asynchronous soft labials (pi to psi, bi to bzi) for Kurpie. Character n-grams
  can see these; a whole-word lexicon cannot.
- **Speaker idiolect.** Each speaker gets a personal filler vocabulary. A model
  can partly re-identify the speaker. This is the leakage that speaker-disjoint
  splitting must defend against.

## The knob that matters: `marker_dropout`

`marker_dropout` (default `0.20`) is the fraction of samples with no lexical
marker. Real dialect utterances often carry no diagnostic lexeme. This knob
mirrors that. It gives the task an error floor.

Set it to `0.0` and every linear model scores `1.000`. That is a benchmark that
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
tulip train configs/synthetic_family.yaml   # family level, incl. the standard class
```

To write an auditable copy to disk (a JSONL manifest, byte-identical for a given
seed):

```bash
tulip data synthesize --out data/raw/synthetic --seed 7
```

## Audio parity

The synthetic path is not text-only. An audio fixture mirrors the same flow, so
audio pipelines run offline too. Its scores carry the same caveat: a fixture,
not real speech.

## Under the hood

The generator is part of the public data API. `generate_corpus` and
`SyntheticSpec` produce the samples. `write_synthetic_manifest` writes them to
disk. See the [data reference](../reference/data.md).

## Learn more

For the real corpora, the manifest format, speaker-ID surrogates, and the
reproducible-split machinery, see [Datasets](../datasets.md).
