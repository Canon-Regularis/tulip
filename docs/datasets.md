# Datasets: acquisition and local layout

tulip never scrapes web pages at runtime. Every corpus is acquired locally
(respecting its licence) and read from a documented directory under
`data/raw/<name>/`. Start with:

```bash
tulip data download --all
```

which fetches every corpus that has a licence-clean automatic source —
today **NKJP** (GPL tarball, parsed to a manifest), **Common Voice PL**
(text/metadata TSV from a mirror of the CC0 release), and **BIGOS**
(Hugging Face Hub; gated, so accept its conditions and authenticate with
`hf auth login` or `HF_TOKEN` first) — and
prints the exact manual steps for the rest; a failing download never aborts
the remaining corpora — most dialect corpora have no
bulk download, so the standard workflow remains: obtain the material,
assemble a **manifest** (one row per sample), and let the loaders do the
rest. `tulip data list` shows what tulip can find locally.

## The manifest format

A manifest is a CSV, TSV, or JSON Lines file — default name
`manifest.csv` / `manifest.tsv` / `manifest.jsonl` (probed in that order) —
with one row per sample. All columns are optional except that at least one of
`text` / `audio_path` must be present:

| Column | Meaning |
| --- | --- |
| `id` | Stable sample ID (synthesised from file + line when absent) |
| `text` | The transcription or written text |
| `audio_path` | Audio file path, relative to the corpus directory |
| `speaker_id` | Speaker identifier — **fill this whenever the corpus provides it** |
| `family` | Dialect family (`greater_polish`, `lesser_polish`, `masovian`, `silesian`, `kashubian`, `standard`) |
| `dialect` | Regional dialect (e.g. `podhale`, `silesia`, `kurpie`); the family is derived automatically |
| `region` / `village` / `voivodeship` | Finer-grained geography |

Any further columns are preserved in `Sample.metadata`. Example:

```csv
id,text,audio_path,speaker_id,dialect,village
d001,"Hej, baca się pyto, kaj się owce pasą.",clips/d001.wav,inf-07,podhale,Chochołów
d002,"Jo żech je z Katowic i godom po naszymu.",clips/d002.wav,inf-12,silesia,Katowice
```

**Speaker IDs and leakage.** Splits are speaker-disjoint: no speaker appears
in more than one of train/validation/test. When a manifest has no
`speaker_id`, a stable surrogate is synthesised from the sample's
village/region/dialect metadata (grouping errs toward *over*-grouping, the
safe direction). Explicit speaker IDs always beat surrogates — record them
when the source provides any.

## Tier 1 — dialect corpora with fine-grained geography

### dialektarium — <https://dialektarium.pl/>

Recordings of dialectal Polish with aligned transcriptions and per-sample
village/region metadata. No bulk download: export or transcribe the material
you are licensed to use into:

```text
data/raw/dialektarium/
    manifest.csv        # text, audio_path, speaker_id, dialect, region, village
    clips/*.wav
```

### dgp — Dialekty i gwary polskie. Kompendium internetowe

<https://przewodnik.tmjp.pl/dgp-dialekty-i-gwary-polskie-kompendium-internetowe/>

Curated dialect text samples organised by dialect group and region. Assemble
the texts you may use into `data/raw/dgp/manifest.csv` with `text`,
`dialect`, `region`, and `village` columns.

## Tier 2 — single-dialect corpora

### korpus_spiski — <https://journals.akademicka.pl/lv/article/view/727>

Transcribed spoken Spisz dialect. Layout: `data/raw/korpus_spiski/manifest.csv`.
The loader defaults every row to `dialect=spisz`; provide `village` and
`speaker_id` where known.

### mackowce — Elektroniczny Korpus Tekstów Gwarowych z Maćkowiec

<https://przewodnik.tmjp.pl/ektgm-elektroniczny-korpus-tekstow-gwarowych-z-mackowiec-na-podolu/>

Borderland (Podolia) dialect texts. Layout: `data/raw/mackowce/manifest.csv`;
rows default to `dialect=podolia`.

## Tier 3 — general Polish and weakly labelled speech

### nkjp — Narodowy Korpus Języka Polskiego — <https://nkjp.pl/>

Standard-Polish negatives for dialect-vs-standard classification.
**Automatic**: `tulip data download nkjp` streams the NKJP-1M balanced
subcorpus tarball (~163 MB, GNU GPL, from `clip.ipipan.waw.pl`), parses its
TEI `text.xml` documents in memory, and writes ~40k paragraphs to
`data/raw/nkjp/manifest.csv` — one surrogate speaker per source document, so
splits stay leakage-free. Rows are labelled `family=standard` automatically.
Manual assembly with the same layout (a `text` column) also works.

### spokes — <https://spokes.clarin-pl.eu/>

Conversational spoken Polish (predominantly standard). Layout:
`data/raw/spokes/manifest.csv` with `text` and `speaker_id`.

### common_voice_pl — <https://commonvoice.mozilla.org/>

Uses the **official release layout directly** — no manifest needed:

```text
data/raw/common_voice_pl/
    validated.tsv       # or pass tsv="train.tsv" etc. in the dataset params
    clips/*.mp3
```

**Automatic (text only)**: `tulip data download common_voice_pl` fetches
`validated.tsv` (sentences + speaker/accent metadata, CC0) from a community
mirror of the release — Mozilla's portal is email-gated and the official
Hub repo is a script-era dataset modern `datasets` cannot load. Audio clips
are deliberately not fetched (tens of GB); for audio experiments download
the official release from Mozilla and drop `clips/` next to the TSV.

`client_id` becomes the speaker ID. Rows default to standard Polish;
self-reported accent strings are kept in metadata and can be promoted to
dialect labels explicitly:

```yaml
datasets:
  - name: common_voice_pl
    params:
      accent_to_dialect:
        śląski: silesia
```

## Tier 4 — ASR aggregations

### bigos — <https://huggingface.co/datasets/michaljunczyk/pl-asr-bigos>

Aggregated Polish ASR corpora. `tulip data download bigos` fetches the
transcriptions automatically, but the dataset is **gated**: sign in on the
Hub, accept the access conditions on the dataset page, and authenticate
locally (`hf auth login`, or set `HF_TOKEN`) before running it.
Alternatively assemble `data/raw/bigos/manifest.csv` yourself, or stream
transcriptions (text-only) in configs with the `hf` extra:

```yaml
datasets:
  - name: bigos
    params:
      from_hub: true
      split: train
      limit: 10000
```

## The `synthetic` reference corpus — no acquisition required

Every corpus above needs a licence and a download. `synthetic` needs neither: it
is **generated in-process**, so a fresh clone can run the whole toolkit
end to end:

```bash
tulip train configs/synthetic_text.yaml     # dialect level
tulip train configs/synthetic_family.yaml   # family level, incl. the `standard` class
```

It is a **benchmark fixture, not real speech**. Scores on it say nothing about
real-world dialect identification. What it does provide is a *reproducible,
learnable* task, grounded in the project's own linguistic resources:

- **Lexical signal** — genuine marker lexemes drawn from
  `src/tulip/features/text/lexicons/dialect_markers.yaml` (Podhale *baca*,
  Silesian *gryfny*, Kashubian *chëcz*, …).
- **Phonological signal** — real sound changes applied as deterministic string
  transforms: *mazurzenie* (cz/sz/ż → c/s/z) for the Masovian group, and
  asynchronous soft-labials (pi → psi, bi → bzi) for Kurpie. These are what
  character n-grams can see and a whole-word lexicon cannot.
- **Speaker idiolect** — each speaker gets a personal filler vocabulary, so a
  model can partly re-identify the speaker. That is exactly the leakage
  speaker-disjoint splitting must defend against, so the split step is
  genuinely exercised rather than trivially satisfied.

The knob that matters is **`marker_dropout`** (default `0.20`): the fraction of
samples carrying *no* lexical marker at all, mirroring the fact that plenty of
real dialect utterances contain no diagnostic lexeme. It gives the task an
irreducible error floor. Set it to `0.0` and every linear model scores a perfect
1.000 — a benchmark that cannot rank anything. Generator parameters are passed
through `params:` like any other loader:

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

To materialise an auditable copy on disk (a JSONL manifest, byte-identical for a
given seed):

```bash
tulip data synthesize --out data/raw/synthetic --seed 7
```

## The `synthetic_audio` corpus — audio parity, no acquisition

The audio analogue of `synthetic`. A deterministic source-filter synthesiser
writes short 16 kHz mono WAV clips whose **pitch register (F0)**, **vowel-space
formants (F1/F2/F3)**, and **spectral tilt** are correlated with the dialect
class, so the classical audio features (`mfcc`, `pitch`, `formants`,
`spectral_centroid`) separate the classes and the whole audio path runs
end-to-end with no data acquisition:

```bash
tulip train configs/synthetic_audio.yaml       # generate clips, split, train, evaluate
tulip data synthesize-audio --out data/raw/synthetic_audio --seed 7   # materialise an auditable copy
```

Clips are written with the stdlib `wave` module (int16, no dithering), so
*generation* needs no audio extra; *reading* them (the feature extractors) needs
the `audio` extra (`librosa`/`soundfile`/`parselmouth`). Neural speech models
(`wav2vec2`, `ecapa_tdnn`, …) need the heavier `speech` extra and are not
exercised by this fixture. Like `synthetic`, it is a **benchmark fixture, not
real speech** — scores on it say nothing about real dialect identification.

## Validating a manifest

Before trusting a manifest you assembled by hand, check it:

```bash
tulip data validate data/raw/<corpus>/manifest.csv
```

It reports structural errors (no `text`/`audio_path` column, malformed rows, bad
encoding), missing audio files, and whether surrogate `speaker_id`s will be
synthesised — which changes how speaker-disjoint splitting groups your data.
Labels outside the taxonomy are **warnings, not errors**: corpus-specific label
strings are explicitly allowed to flow through. The command exits non-zero only
on errors, so it works as a CI gate.

## Building a reproducible benchmark split

```bash
tulip data prepare configs/text_baseline.yaml
```

writes `train/validation/test.jsonl` plus `build_manifest.json` — sizes,
per-class distribution, source counts, and the exact cleaning/dedup/split
configuration — under the experiment's artifact directory. Publishing that
directory (where licences allow) is what makes results comparable:
deduplication runs **before** splitting so near-duplicates can never straddle
splits, and grouping guarantees speaker disjointness. `tulip benchmark`
then evaluates any number of models against the identical frozen split.

For a whole suite of experiments at once, `tulip leaderboard benchmarks/suite.yaml`
regenerates the committed leaderboard under `benchmarks/results/`. Its
`leaderboard.md` and `provenance.json` are byte-identical across re-runs for a
fixed seed — see `benchmarks/README.md`.
