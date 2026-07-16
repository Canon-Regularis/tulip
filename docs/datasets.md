# Datasets: acquisition and local layout

tulip never scrapes web pages at runtime. You acquire each corpus locally, under
its licence, and place it under `data/raw/<name>/`. Start with:

```bash
tulip data download --all
```

This fetches every corpus that has a licence-clean automatic source. Today that
means:

- **NKJP.** GPL tarball, parsed to a manifest.
- **Common Voice PL.** Text and metadata TSV from a CC0 mirror.
- **BIGOS.** From the Hugging Face Hub. It is gated, so accept its conditions and
  authenticate first (`hf auth login` or `HF_TOKEN`).

The command prints the exact manual steps for every other corpus. A failing
download does not stop the others. Most dialect corpora have no bulk download, so
the usual path is: get the material, assemble a manifest (one row per sample),
and let the loaders read it. `tulip data list` shows what tulip can find locally.

Two flags shape a fetch. `--limit N` caps how much each loader pulls, which is
what makes a smoke run tractable. `--audio` additionally materialises audio clips
for the loaders that can stream them (`bigos` and `common_voice_pl`); it is a
no-op for the rest, so `--all --audio` still fetches everyone's text. The full
audio corpora run to tens of GB, so always pair `--audio` with `--limit`:

```bash
tulip data download bigos --audio --limit 500
```

Because a corpus already present on disk is skipped, add `--force` to re-fetch a
text-only corpus with its audio.

## The manifest format

A manifest is a CSV, TSV, or JSON Lines file. The default names are
`manifest.csv`, `manifest.tsv`, and `manifest.jsonl`, probed in that order. Each
row is one sample. Every column is optional, except that each row needs at least
one of `text` or `audio_path`.

| Column | Meaning |
| --- | --- |
| `id` | Stable sample ID (synthesised from file and line when absent) |
| `text` | The transcription or written text |
| `audio_path` | Audio file path, relative to the corpus directory |
| `speaker_id` | Speaker identifier. Fill this whenever the corpus provides it. |
| `family` | Dialect family (`greater_polish`, `lesser_polish`, `masovian`, `silesian`, `kashubian`, `standard`) |
| `dialect` | Regional dialect (e.g. `podhale`, `silesia`, `kurpie`). The family is derived. |
| `region` / `village` / `voivodeship` | Finer geography |

Any other columns are kept in `Sample.metadata`. Example:

```csv
id,text,audio_path,speaker_id,dialect,village
d001,"Hej, baca się pyto, kaj się owce pasą.",clips/d001.wav,inf-07,podhale,Chochołów
d002,"Jo żech je z Katowic i godom po naszymu.",clips/d002.wav,inf-12,silesia,Katowice
```

**Speaker IDs and leakage.** Splits are speaker-disjoint. No speaker appears in
more than one of train, validation, or test. When a manifest has no `speaker_id`,
tulip synthesises a stable surrogate from the village, region, and dialect
metadata. This grouping errs toward over-grouping, which is the safe direction.
Explicit speaker IDs always beat surrogates, so record them when you can.

## Tier 1: dialect corpora with fine-grained geography

### dialektarium (<https://dialektarium.pl/>)

Recordings of dialectal Polish with aligned transcriptions and per-sample
village and region metadata. There is no bulk download. Export or transcribe the
material you are licensed to use into:

```text
data/raw/dialektarium/
    manifest.csv        # text, audio_path, speaker_id, dialect, region, village
    clips/*.wav
```

### dgp: Dialekty i gwary polskie. Kompendium internetowe

<https://przewodnik.tmjp.pl/dgp-dialekty-i-gwary-polskie-kompendium-internetowe/>

Curated dialect text samples by dialect group and region. Assemble the texts you
may use into `data/raw/dgp/manifest.csv` with `text`, `dialect`, `region`, and
`village` columns.

## Tier 2: single-dialect corpora

### korpus_spiski (<https://journals.akademicka.pl/lv/article/view/727>)

Transcribed spoken Spisz dialect. Layout: `data/raw/korpus_spiski/manifest.csv`.
The loader defaults every row to `dialect=spisz`. Provide `village` and
`speaker_id` where known.

### mackowce: Elektroniczny Korpus Tekstów Gwarowych z Maćkowiec

<https://przewodnik.tmjp.pl/ektgm-elektroniczny-korpus-tekstow-gwarowych-z-mackowiec-na-podolu/>

Borderland (Podolia) dialect texts. Layout: `data/raw/mackowce/manifest.csv`.
Rows default to `dialect=podolia`.

## Tier 3: general Polish and weakly labelled speech

### nkjp: Narodowy Korpus Języka Polskiego (<https://nkjp.pl/>)

Standard-Polish negatives for dialect-vs-standard classification.
`tulip data download nkjp` streams the NKJP-1M balanced subcorpus tarball (about
163 MB, GNU GPL, from `clip.ipipan.waw.pl`). It parses the TEI `text.xml`
documents in memory and writes about 40k paragraphs to
`data/raw/nkjp/manifest.csv`. Each source document becomes one surrogate speaker,
so splits stay leakage-free. Rows are labelled `family=standard`. Manual assembly
with the same layout (a `text` column) also works.

### spokes (<https://spokes.clarin-pl.eu/>)

Conversational spoken Polish, mostly standard. Layout:
`data/raw/spokes/manifest.csv` with `text` and `speaker_id`.

### common_voice_pl (<https://commonvoice.mozilla.org/>)

This loader reads the official release layout directly. No manifest is needed:

```text
data/raw/common_voice_pl/
    validated.tsv       # or pass tsv="train.tsv" etc. in the dataset params
    clips/*.mp3
```

`tulip data download common_voice_pl` fetches `validated.tsv` (sentences plus
speaker and accent metadata, CC0) from a community mirror. By default it does not
fetch the audio clips, which run to tens of GB.

Add `--audio` to stream the clips from the mirror with the `datasets` library
(the `hf` extra). It writes each clip under `clips/` and a matching release TSV,
so the standard loader then reads text and audio together with no extra setup.
Pair it with `--limit` to fetch a tractable slice, and `--force` if a text-only
`validated.tsv` is already present:

```bash
tulip data download common_voice_pl --audio --limit 500 --force
```

Alternatively, download the official release from Mozilla and drop `clips/` next
to the TSV.

`client_id` becomes the speaker ID. Rows default to standard Polish.
Self-reported accent strings are kept in metadata. You can promote them to
dialect labels:

```yaml
datasets:
  - name: common_voice_pl
    params:
      accent_to_dialect:
        śląski: silesia
```

## Tier 4: ASR aggregations

### bigos (<https://huggingface.co/datasets/michaljunczyk/pl-asr-bigos>)

Aggregated Polish ASR corpora. `tulip data download bigos` fetches the
transcriptions into `data/raw/bigos/manifest.csv`. The dataset is gated: sign in
on the Hub, accept the access conditions on the dataset page, and authenticate
locally (`hf auth login`, or set `HF_TOKEN`) first.

Add `--audio` to also materialise the clips: each streamed record's audio is
written under `clips/` and recorded in the manifest's `audio_path` column, which
is what gives the neural-audio models a first-party real corpus. Pair it with
`--limit`, the full corpus is tens of GB:

```bash
tulip data download bigos --audio --limit 500
```

You can also assemble `data/raw/bigos/manifest.csv` yourself, or stream
transcriptions in configs with the `hf` extra:

```yaml
datasets:
  - name: bigos
    params:
      from_hub: true
      split: train
      limit: 10000
```

## The synthetic corpora: no acquisition required

Two corpora need no licence and no download, because they are generated in
memory: `synthetic` (text) and `synthetic_audio` (audio). A fresh clone can run
the whole toolkit offline.

```bash
tulip train configs/synthetic_text.yaml     # dialect level
tulip train configs/synthetic_audio.yaml    # audio parity
```

Both are test fixtures, not real speech. Their scores say nothing about real
dialect identification. For what they contain, the `marker_dropout` knob, and how
to write an auditable copy to disk, see
[Synthetic corpora](guide/synthetic-corpora.md).

## Transcribing audio to text: the ASR bridge

An audio corpus can feed the text pipeline. `tulip data transcribe` runs each
clip through Whisper and writes a manifest whose `text` is the transcript, so a
speech corpus (a `--audio` fetch of BIGOS, say) becomes a transcribed-speech
track that every text model trains on directly. The clips' labels and audio
paths are preserved, so the same corpus stays usable for the audio models too.

```bash
tulip data transcribe data/raw/bigos/manifest.csv --out data/raw/bigos-asr --limit 500
```

It reads a split `.jsonl`, a manifest, or a directory, and skips samples with no
audio. Transcription needs the `speech` extra (Whisper via `transformers`). Pass
`--cache <dir>` to memoise transcripts by audio content and checkpoint, so a
rerun is offline and free; `--checkpoint` and `--language` (default `pl`) pick
the model and force the decode language. The result is an ordinary manifest, so
validate and train on it like any other corpus.

## Validating a manifest

Check a manifest you assembled by hand before you trust it:

```bash
tulip data validate data/raw/<corpus>/manifest.csv
```

It reports structural errors (no `text` or `audio_path` column, malformed rows,
bad encoding), missing audio files, and whether surrogate `speaker_id`s will be
synthesised. Labels outside the taxonomy are warnings, not errors: corpus-specific
label strings are allowed to flow through. The command exits non-zero only on
errors, so it works as a CI gate.

## Building a reproducible benchmark split

```bash
tulip data prepare configs/text_baseline.yaml
```

This writes `train/validation/test.jsonl` plus `build_manifest.json` (sizes,
per-class distribution, source counts, and the exact configuration) under the
experiment's artifact directory. Two things make the result comparable across
runs. Deduplication runs before splitting, so near-duplicates cannot straddle
splits. Grouping keeps speakers disjoint. `tulip benchmark` then evaluates any
number of models against the same frozen split.

For a whole suite at once, `tulip leaderboard benchmarks/suite.yaml` regenerates
the committed leaderboard under `benchmarks/results/`. Its `leaderboard.md` and
`provenance.json` are byte-identical across re-runs for a fixed seed. See
`benchmarks/README.md`.

That committed board runs on the offline `synthetic` corpus. Its scientific
companion, `benchmarks/real_text_suite.yaml`, runs the same competitors on real
prose (`dialektarium` + `dgp`). Acquire those two corpora as above first, then:

```bash
tulip leaderboard benchmarks/real_text_suite.yaml --out artifacts/real-text
```

Nothing from the real-text suite is committed, because its inputs are not in the
repository, so it makes no byte-for-byte reproducibility claim. See
`benchmarks/README.md`.
