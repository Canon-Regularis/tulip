# Datasets: acquisition and local layout

tulip never scrapes remote sources at runtime. Every corpus is acquired
locally (respecting its licence) and read from a documented directory under
`data/raw/<name>/`. Most corpora have no bulk download, so the standard
workflow is: obtain the material, assemble a **manifest** (one row per
sample), and let the loaders do the rest. `tulip data list` shows what tulip
can find locally.

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

Standard-Polish negatives for dialect-vs-standard classification. Export
sentences/paragraphs into `data/raw/nkjp/manifest.csv` with a `text` column;
rows are labelled `family=standard` automatically.

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

Aggregated Polish ASR corpora. Either assemble `data/raw/bigos/manifest.csv`
locally, or stream transcriptions (text-only) from the Hugging Face Hub with
the `hf` extra:

```yaml
datasets:
  - name: bigos
    params:
      from_hub: true
      split: train
      limit: 10000
```

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
