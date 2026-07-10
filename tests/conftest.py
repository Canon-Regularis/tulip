"""Shared fixtures, helpers, and corpus builders for the tulip test suite."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from tulip.config.schemas import ComponentConfig, DataConfig, ExperimentConfig, SplitConfig
from tulip.core.types import DialectLabels, Sample

#: Records shaped like the BIGOS Hub schema (see loaders/bigos.py field probes).
BIGOS_HUB_RECORDS: list[dict[str, str]] = [
    {"ref_orig": "Pierwsze zdanie testowe.", "speaker_id": "spk-1", "dataset": "sub-a"},
    {"ref_orig": "Drugie zdanie testowe.", "speaker_id": "spk-2", "dataset": "sub-a"},
    {"ref_orig": "", "speaker_id": "spk-3", "dataset": "sub-a"},  # empty text: skipped
    {"ref_orig": "Trzecie, z przecinkiem w tekście.", "dataset": "sub-b"},  # no speaker
]


@pytest.fixture
def fake_bigos_hub(monkeypatch: pytest.MonkeyPatch):
    """Install a stub ``datasets`` module streaming BIGOS_HUB_RECORDS.

    Guarantees hub-touching tests never reach the network, even on machines
    where the real ``datasets`` library is installed.
    """
    import sys
    from types import ModuleType, SimpleNamespace

    calls = SimpleNamespace(load_dataset_args=None)

    def load_dataset(name, config, *, split, streaming):
        calls.load_dataset_args = (name, config, split, streaming)
        return iter(BIGOS_HUB_RECORDS)

    module = ModuleType("datasets")
    module.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", module)
    return calls


def block_imports(monkeypatch: pytest.MonkeyPatch, *blocked: str) -> None:
    """Make ``importlib.import_module`` fail for the given module trees.

    Lets optional-dependency failure paths be exercised even on machines where
    the dependency IS installed.
    """
    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if any(name == root or name.startswith(root + ".") for root in blocked):
            raise ImportError(f"blocked for test: {name}")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)


# Tiny synthetic corpus: crude dialect-flavoured Polish, multiple speakers per
# dialect so speaker-disjoint splitting is exercisable. Not linguistically
# faithful; only the statistical shape matters for tests.
_DIALECT_SENTENCES: dict[str, list[str]] = {
    "podhale": [
        "Hej, baca się pyto, kaj się owce pasą na holi.",
        "Juhas poseł na grań i widzioł cołkiem piykne hole.",
        "Kie góral idzie ku dolinie, to se śpiywo po naszymu.",
        "Baca wzion ciupagę i poseł ku sałasowi na wiyrchu.",
    ],
    "silesia": [
        "Jo żech je z Katowic i godom po naszymu cołki czos.",
        "Kaj żeś boł wczorej, bo cie żech nie widzioł na placu?",
        "Dej pozór na bajtla, bo lecie po drodze ku familokowi.",
        "Niy ma co godać, trza robić, pado starzik.",
    ],
    "kurpie": [
        "U nos w boru to psiwo warzą jesce po staremu.",
        "Chłopoki poślo do lasu na jagody i grziby zbzierać.",
        "Kobziety śpsiewajo w kościele w niedziele rano.",
        "Na Kurpsiach bursztyn kopalo się od dawna w psiochach.",
    ],
    "standard": [
        "Wczoraj pojechałem do Warszawy na spotkanie służbowe.",
        "Prognoza pogody zapowiada deszcz w całym kraju.",
        "Nowa ustawa wejdzie w życie od pierwszego stycznia.",
        "Dzieci wróciły ze szkoły i odrobiły lekcje przed kolacją.",
    ],
}


def make_samples(*, repeats: int = 3) -> list[Sample]:
    """Build a small deterministic corpus with several speakers per dialect."""
    samples: list[Sample] = []
    for dialect, sentences in _DIALECT_SENTENCES.items():
        for speaker_index in range(repeats):
            speaker = f"{dialect}-spk{speaker_index}"
            for sentence_index, sentence in enumerate(sentences):
                samples.append(
                    Sample(
                        id=f"{speaker}-{sentence_index}",
                        text=f"{sentence} (wariant {speaker_index})",
                        speaker_id=speaker,
                        labels=DialectLabels(
                            dialect=dialect if dialect != "standard" else None,
                            family="standard" if dialect == "standard" else None,
                        ),
                        source="synthetic",
                    )
                )
    return samples


@pytest.fixture
def synthetic_samples() -> list[Sample]:
    """A small labelled corpus spanning four classes and twelve speakers."""
    return make_samples()


@pytest.fixture
def synthetic_texts_and_labels(synthetic_samples: list[Sample]) -> tuple[list[str], list[str]]:
    """Parallel lists of texts and dialect-or-standard labels."""
    texts = [s.text or "" for s in synthetic_samples]
    labels = [s.labels.dialect or s.labels.family or "unknown" for s in synthetic_samples]
    return texts, labels


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded numpy random generator for deterministic tests."""
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def trained_text_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small trained + saved text classifier, shared by the CLI/serve tests.

    Session-scoped: training even this tiny model dominates those tests'
    runtime, and every consumer only reads the artifact.
    """
    from tulip.pipeline import DialectClassifier

    artifact = tmp_path_factory.mktemp("trained-model") / "model"
    classifier = DialectClassifier(model="logistic_regression", features=["char_tfidf"], seed=42)
    classifier.fit(make_samples())
    classifier.save(artifact)
    return artifact


# Deliberately comma-free sentence templates: corpus builders below write
# naive (unquoted) CSV rows.
DIALECT_TEMPLATES: dict[str, str] = {
    "podhale": "Hej baca się pyto kaj się owce pasą na holi wariant {i}.",
    "silesia": "Jo żech je z Katowic i godom po naszymu cołki czos wariant {i}.",
    "kurpie": "U nos w boru psiwo warzą jesce po staremu wariant {i}.",
}


def write_manifest_corpus(
    directory: Path,
    *,
    speakers: int = 5,
    variants: int = 3,
    extra_rows: tuple[str, ...] = (),
) -> Path:
    """Write a small on-disk CSV manifest corpus and return its directory.

    Produces ``len(DIALECT_TEMPLATES) * speakers * variants`` labelled rows
    with several speakers per dialect, so speaker-disjoint splitting is
    exercisable. ``extra_rows`` appends raw CSV lines (e.g. duplicates or
    degenerate rows) for tests that need them.
    """
    rows = ["id,text,speaker_id,dialect"]
    for dialect, template in DIALECT_TEMPLATES.items():
        for speaker in range(speakers):
            for i in range(variants):
                text = template.format(i=f"{speaker}-{i}")
                rows.append(f"{dialect}-{speaker}-{i},{text},{dialect}-spk{speaker},{dialect}")
    rows.extend(extra_rows)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return directory


def make_manifest_experiment_config(
    corpus: Path,
    output_dir: Path,
    *,
    name: str = "mini-text",
    **overrides: object,
) -> ExperimentConfig:
    """Build a complete, runnable text-experiment config over a manifest corpus.

    Deduplication is off because the corpus's "wariant N" texts are
    intentionally near-identical; keyword ``overrides`` replace any
    ExperimentConfig field.
    """
    fields: dict[str, object] = {
        "name": name,
        "seed": 42,
        "data": DataConfig(
            datasets=[ComponentConfig(name="manifest", params={"root": str(corpus)})],
            root=corpus.parent,
            deduplicate=False,
            min_text_chars=10,
        ),
        "features": [ComponentConfig(name="char_tfidf")],
        "model": ComponentConfig(name="logistic_regression"),
        "split": SplitConfig(seed=42),
        "output_dir": output_dir,
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)  # type: ignore[arg-type]
