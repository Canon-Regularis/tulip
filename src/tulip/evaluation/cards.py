"""Deterministic markdown dataset and model cards.

These helpers render human-readable *cards* from the artifacts the toolkit
already writes -- the ``build_manifest.json`` of a dataset build and the
``metadata.json`` sidecar of a saved model -- so a published benchmark ships
with committable, diff-friendly documentation of exactly what was built and
trained.

Two properties are load-bearing and deliberately enforced:

* **No recomputation.** :func:`dataset_card` reuses the counts, class
  distribution, and sources already recorded in the manifest rather than
  re-deriving them from the corpus; only :func:`dataset_card_from_splits`
  computes (from in-memory splits) the few facts a manifest omits, notably
  speaker counts.
* **Byte-stability.** The same inputs always render the identical string --
  no timestamps, no dependence on ``dict`` iteration order (every key set is
  sorted explicitly) -- because these cards are committed to the repository
  and must diff cleanly.

Missing or absent optional fields degrade to ``"n/a"`` and never raise: a
partial artifact still produces a usable card.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from tulip.evaluation._format import format_metric, markdown_table
from tulip.labels.taxonomy import LabelLevel, display_name

if TYPE_CHECKING:
    from tulip.core.types import DatasetInfo
    from tulip.data.splitting import DatasetSplits
    from tulip.evaluation.report import EvaluationReport

__all__ = ["dataset_card", "dataset_card_from_splits", "model_card"]

#: Sentinel the builder/splitter records for samples with no label at a level.
_UNLABELLED = "__unlabelled__"
#: Canonical split ordering; any other split names sort after these.
_CANONICAL_SPLITS = ("train", "validation", "test")
#: Placeholder for values that are absent from a (possibly partial) artifact.
_NA = "n/a"


# --------------------------------------------------------------------- helpers


def _na(value: Any) -> str:
    """Render ``value`` for display, mapping ``None``/blank to ``"n/a"``."""
    if value is None:
        return _NA
    text = str(value).strip()
    return text if text else _NA


def _as_mapping(value: Any) -> Mapping[str, Any]:
    """Return ``value`` if it is a mapping, else an empty mapping (never raise)."""
    return value if isinstance(value, Mapping) else {}


def _humanize_label(label: str) -> str:
    """Humanise a class label, rendering the unlabelled sentinel readably."""
    if label == _UNLABELLED:
        return "(unlabelled)"
    return display_name(label)


def _ordered_splits(*mappings: Mapping[str, Any]) -> list[str]:
    """Union the keys of ``mappings`` into canonical (train/val/test) order.

    Split names outside the canonical trio sort lexicographically after it, so
    the ordering is total and deterministic regardless of input ``dict`` order.
    """
    seen: set[str] = set()
    for mapping in mappings:
        seen.update(_as_mapping(mapping))
    ordered = [name for name in _CANONICAL_SPLITS if name in seen]
    ordered.extend(sorted(seen.difference(_CANONICAL_SPLITS)))
    return ordered


def _format_value(value: Any) -> str:
    """Format a config value deterministically for a parameter table."""
    if value is None:
        return _NA
    if isinstance(value, bool):  # bool is an int subclass; check it first
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return str(value)
    import json

    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _format_params(params: Any) -> str:
    """Render a component's params as a compact, key-sorted ``k=v; ...`` string."""
    params = _as_mapping(params)
    if not params:
        return "—"
    return "; ".join(f"{key}={_format_value(params[key])}" for key in sorted(params))


# -------------------------------------------------------------- dataset cards


def dataset_card(info: DatasetInfo, build_manifest: Mapping[str, Any]) -> str:
    """Render a dataset card from a corpus's static info and its build manifest.

    Reuses the sizes, per-label class distribution, and source counts already
    written to ``build_manifest.json`` (see :class:`tulip.data.DatasetBuilder`);
    nothing is recomputed. Speaker counts are not in the manifest and therefore
    render as ``"n/a"`` -- use :func:`dataset_card_from_splits` when they matter.

    Args:
        info: Static metadata for the corpus (name, license, label levels, ...).
        build_manifest: The parsed ``build_manifest.json`` mapping. Any missing
            key degrades gracefully rather than raising.

    Returns:
        A byte-stable markdown document (no trailing newline).
    """
    label_level = str(build_manifest.get("label_level") or LabelLevel.DIALECT.value)
    return _render_dataset_card(
        info,
        sizes=_as_mapping(build_manifest.get("sizes")),
        total=build_manifest.get("total"),
        label_level=label_level,
        class_distribution=_as_mapping(build_manifest.get("class_distribution")),
        sources=_as_mapping(build_manifest.get("sources")),
        speaker_counts=None,
    )


def dataset_card_from_splits(
    info: DatasetInfo,
    splits: DatasetSplits,
    *,
    level: LabelLevel = LabelLevel.DIALECT,
) -> str:
    """Render a dataset card by computing counts directly from in-memory splits.

    Unlike :func:`dataset_card`, this derives the per-label class distribution,
    source counts, and -- crucially -- the distinct-speaker count per split from
    the actual samples, so the card documents leakage-relevant speaker coverage.

    Args:
        info: Static metadata for the corpus.
        splits: The train/validation/test partitions to summarise.
        level: Granularity at which to tabulate the class distribution. Defaults
            to :attr:`LabelLevel.DIALECT`, matching the builder's manifest.

    Returns:
        A byte-stable markdown document (no trailing newline).
    """
    sizes: dict[str, int] = {}
    distribution: dict[str, dict[str, int]] = {}
    speaker_counts: dict[str, int] = {}
    sources: Counter[str] = Counter()
    for name, samples in splits.as_dict().items():
        sizes[name] = len(samples)
        distribution[name] = dict(
            Counter(sample.labels.at_level(level) or _UNLABELLED for sample in samples)
        )
        speaker_counts[name] = len({s.speaker_id for s in samples if s.speaker_id})
        sources.update(sample.source for sample in samples)
    return _render_dataset_card(
        info,
        sizes=sizes,
        total=splits.total,
        label_level=level.value,
        class_distribution=distribution,
        sources=dict(sources),
        speaker_counts=speaker_counts,
    )


def _render_dataset_card(
    info: DatasetInfo,
    *,
    sizes: Mapping[str, Any],
    total: Any,
    label_level: str,
    class_distribution: Mapping[str, Any],
    sources: Mapping[str, Any] | None,
    speaker_counts: Mapping[str, int] | None,
) -> str:
    """Assemble the dataset card from already-extracted counts (shared renderer)."""
    parts = [f"# Dataset card — {_na(info.name)}"]
    if str(info.description).strip():
        parts.append(str(info.description).strip())
    parts.append(_dataset_overview(info))
    parts.append(_splits_section(sizes, total, speaker_counts))
    parts.append(_class_distribution_section(label_level, class_distribution))
    parts.append(_sources_section(sources))
    return "\n\n".join(parts)


def _dataset_overview(info: DatasetInfo) -> str:
    """Render the header bullet list from a :class:`DatasetInfo`."""
    tasks = ", ".join(info.tasks) if info.tasks else _NA
    levels = ", ".join(level.value for level in info.label_levels) if info.label_levels else _NA
    contents = ", ".join(info.contents) if info.contents else _NA
    bullets = [
        ("Name", info.name),
        ("URL", _na(info.url)),
        ("Tier", info.tier),
        ("Tasks", tasks),
        ("Contents", contents),
        ("Label levels", levels),
        ("License", info.license),
    ]
    return "\n".join(f"- **{key}:** {_na(value)}" for key, value in bullets)


def _splits_section(
    sizes: Mapping[str, Any],
    total: Any,
    speaker_counts: Mapping[str, int] | None,
) -> str:
    """Render the split-sizes table (samples + speakers, with a totals row)."""
    split_names = _ordered_splits(sizes, speaker_counts or {})
    rows: list[tuple[str, str, str]] = []
    sample_total = 0
    speaker_total = 0
    for name in split_names:
        count = sizes.get(name)
        if isinstance(count, int):
            sample_total += count
        speakers = _NA
        if speaker_counts is not None:
            value = speaker_counts.get(name, 0)
            speaker_total += int(value)
            speakers = str(int(value))
        rows.append((name, _na(count), speakers))

    total_samples = int(total) if isinstance(total, int) else sample_total
    total_speakers = str(speaker_total) if speaker_counts is not None else _NA
    rows.append(("**Total**", str(total_samples), total_speakers))
    table = markdown_table(("Split", "Samples", "Speakers"), rows)
    return f"## Splits\n\n{table}"


def _class_distribution_section(
    label_level: str,
    class_distribution: Mapping[str, Any],
) -> str:
    """Render one row per label with per-split counts and a total column."""
    heading = f"## Class distribution ({label_level})"
    split_names = _ordered_splits(class_distribution)
    per_split = {name: _as_mapping(class_distribution.get(name)) for name in split_names}

    labels: set[str] = set()
    for counts in per_split.values():
        labels.update(counts)
    if not labels:
        return f"{heading}\n\nNo class distribution recorded."

    rows: list[tuple[str, ...]] = []
    for label in sorted(labels):
        cells = [_humanize_label(label)]
        label_total = 0
        for name in split_names:
            count = int(per_split[name].get(label, 0) or 0)
            label_total += count
            cells.append(str(count))
        cells.append(str(label_total))
        rows.append(tuple(cells))

    headers = ("Label", *split_names, "Total")
    return f"{heading}\n\n{markdown_table(headers, rows)}"


def _sources_section(sources: Mapping[str, Any] | None) -> str:
    """Render the source-corpora table (source name -> sample count)."""
    sources = _as_mapping(sources)
    if not sources:
        return "## Source corpora\n\nNo source corpora recorded."
    rows = [(name, _na(sources.get(name))) for name in sorted(sources)]
    return f"## Source corpora\n\n{markdown_table(('Source', 'Samples'), rows)}"


# ----------------------------------------------------------------- model card


def model_card(sidecar: Mapping[str, Any], reports: Mapping[str, EvaluationReport]) -> str:
    """Render a model card from a saved-model sidecar and its evaluation reports.

    The sidecar is the parsed ``metadata.json`` written by
    :func:`tulip.models.persistence.save_model` (environment versions, model
    class, fitted classes, and the nested ``metadata`` dict a
    :class:`~tulip.pipeline.DialectClassifier` records). ``save_model``
    deliberately omits metrics, so headline metrics come from ``reports``.

    Args:
        sidecar: The parsed ``metadata.json`` mapping.
        reports: Evaluation reports keyed by split name (e.g. ``{"validation":
            ..., "test": ...}``). May be empty.

    Returns:
        A byte-stable markdown document (no trailing newline).
    """
    meta = _as_mapping(sidecar.get("metadata"))
    model_cfg = _as_mapping(meta.get("model"))
    title = model_cfg.get("name") or sidecar.get("model_class") or meta.get("kind") or "model"

    parts = [f"# Model card — {_na(title)}"]
    parts.append(_model_overview(sidecar, meta))
    parts.append(_components_section(meta, model_cfg))
    parts.append(_classes_section(sidecar.get("classes")))
    parts.append(_metrics_section(reports))
    return "\n\n".join(parts)


def _model_overview(sidecar: Mapping[str, Any], meta: Mapping[str, Any]) -> str:
    """Render the model-card header bullets (identity, environment, task setup)."""
    bullets = [
        ("Kind", meta.get("kind")),
        ("Model class", sidecar.get("model_class")),
        ("tulip version", sidecar.get("tulip_version")),
        ("Python version", sidecar.get("python_version")),
        ("Task", meta.get("task")),
        ("Target level", meta.get("target")),
        ("Abstain threshold", meta.get("abstain_threshold")),
        ("Seed", meta.get("seed")),
    ]
    return "\n".join(f"- **{key}:** {_na(value)}" for key, value in bullets)


def _components_section(meta: Mapping[str, Any], model_cfg: Mapping[str, Any]) -> str:
    """Render the model params table and the feature-component table."""
    parts = ["## Components", "### Model"]

    model_name = model_cfg.get("name")
    parts.append(f"- **Name:** {_na(model_name)}")
    model_params = _as_mapping(model_cfg.get("params"))
    if model_params:
        rows = [(key, _format_value(model_params[key])) for key in sorted(model_params)]
        parts.append(markdown_table(("Parameter", "Value"), rows))
    else:
        parts.append("No non-default parameters.")

    parts.append("### Features")
    features = meta.get("features")
    feature_list = list(features) if isinstance(features, (list, tuple)) else []
    if not feature_list:
        parts.append("Raw-input model (no feature components).")
    else:
        rows = [
            (
                _na(_as_mapping(entry).get("name")),
                _format_params(_as_mapping(entry).get("params")),
            )
            for entry in feature_list
        ]
        parts.append(markdown_table(("Component", "Parameters"), rows))
    return "\n\n".join(parts)


def _classes_section(classes: Any) -> str:
    """Render the fitted class list (raw label + humanised name), sorted."""
    if not isinstance(classes, (list, tuple)) or not classes:
        return "## Classes\n\nClasses: n/a (model unfitted or exposes no classes)."
    labels = sorted(str(label) for label in classes)
    lines = [f"Trained on **{len(labels)}** classes:", ""]
    lines.extend(f"- `{label}` — {_humanize_label(label)}" for label in labels)
    return "## Classes\n\n" + "\n".join(lines)


def _metrics_section(reports: Mapping[str, EvaluationReport]) -> str:
    """Render a headline metrics table (splits as columns) plus summary lines."""
    reports = _as_mapping(reports)
    if not reports:
        return "## Metrics\n\nNo evaluation reports available."

    split_names = _ordered_splits(reports)
    metric_rows = [
        ("Samples", lambda r: str(r.n_samples)),
        ("Accuracy", lambda r: format_metric(r.accuracy)),
        ("Balanced accuracy", lambda r: format_metric(r.balanced_accuracy)),
        ("F1 (macro)", lambda r: format_metric(r.f1_macro)),
        ("F1 (weighted)", lambda r: format_metric(r.f1_weighted)),
        ("ROC AUC (macro OVR)", lambda r: format_metric(r.roc_auc_macro_ovr)),
    ]
    rows = [
        (name, *(render(reports[split]) for split in split_names)) for name, render in metric_rows
    ]
    table = markdown_table(("Metric", *split_names), rows)

    summaries = "\n".join(
        f"- **{split}:** {reports[split].summary_line()}" for split in split_names
    )
    return f"## Metrics\n\n{table}\n\n{summaries}"
