"""``tulip doctor``: what can I run right now, and what unblocks the rest?

A fresh install carries only the light core. torch, librosa, shap, and the other
heavy stacks are optional extras, so most of the model, feature, and explainer
catalogue is dormant until the matching extra is installed. This module turns
that into a straight answer: it probes each extra, then walks the registries and
marks every component runnable or blocked, naming the exact ``pip install`` that
would unblock it.

The component-to-extra link is not hardcoded here. Each component declares its
extra once, at registration, through the registry's ``metadata`` channel (for
example ``@MODELS.register("herbert", metadata={"extra": "transformers"})``);
this module only reads it back with :meth:`Registry.metadata`. A component with
no declared extra runs on the core install. Availability is probed with
:func:`tulip.utils.optional.is_available`, the same check the runtime uses, so
the report never disagrees with what an actual run would do.

The report is a plain diagnostic of the live interpreter, not a committed
artifact, so it carries the real Python and platform strings and is not required
to be byte-stable across machines.
"""

from __future__ import annotations

import platform

from pydantic import BaseModel, ConfigDict

from tulip import __version__
from tulip.utils.optional import is_available

__all__ = [
    "ComponentStatus",
    "DoctorReport",
    "ExtraStatus",
    "component_statuses",
    "probe_extras",
    "run_doctor",
]

#: Optional extra -> (import module names that must resolve, one-line purpose).
#:
#: The modules are *import* names, not pip names, because they diverge
#: (``praat-parselmouth`` imports as ``parselmouth``, ``umap-learn`` as
#: ``umap``). This is the one place that mapping lives; the extras themselves are
#: declared in ``pyproject.toml`` and the per-extra module set mirrors it. Only
#: runtime capabilities are listed; ``docs`` and ``dev`` are tooling and have
#: no bearing on what the toolkit can run.
_EXTRAS: dict[str, tuple[tuple[str, ...], str]] = {
    "anthropic": (("anthropic",), "the constrained-choice LLM baseline"),
    "audio": (("librosa", "soundfile", "parselmouth"), "classical audio features"),
    "boosting": (("xgboost", "lightgbm"), "gradient-boosting baselines"),
    "explain": (("shap", "lime"), "SHAP and LIME explanations"),
    "fasttext": (("fasttext",), "the fastText baseline"),
    "hf": (("datasets",), "Hugging Face Hub corpora"),
    "serve": (("fastapi", "uvicorn"), "the HTTP inference service"),
    "speech": (("torch", "torchaudio", "transformers", "speechbrain"), "neural speech models"),
    "transformers": (("torch", "transformers"), "transformer text models"),
    "umap": (("umap",), "UMAP embedding projection"),
    "viz": (("folium", "plotly", "matplotlib"), "maps and charts"),
}


class ExtraStatus(BaseModel):
    """Install state of one optional extra."""

    model_config = ConfigDict(frozen=True)

    name: str
    installed: bool
    purpose: str
    modules: tuple[str, ...]
    missing_modules: tuple[str, ...]

    @property
    def install_hint(self) -> str:
        """The command that installs this extra."""
        return f"pip install 'tulip-dialect[{self.name}]'"


class ComponentStatus(BaseModel):
    """Whether one registry component can run on the current install."""

    model_config = ConfigDict(frozen=True)

    kind: str
    name: str
    extra: str | None
    available: bool


class DoctorReport(BaseModel):
    """A snapshot of the environment, the extras, and the runnable catalogue."""

    model_config = ConfigDict(frozen=True)

    python_version: str
    platform: str
    tulip_version: str
    extras: tuple[ExtraStatus, ...]
    components: tuple[ComponentStatus, ...]

    @property
    def missing_extras(self) -> tuple[ExtraStatus, ...]:
        """Extras that are not installed."""
        return tuple(extra for extra in self.extras if not extra.installed)

    @property
    def blocked_components(self) -> tuple[ComponentStatus, ...]:
        """Components that cannot run until their extra is installed."""
        return tuple(component for component in self.components if not component.available)

    @property
    def runnable_count(self) -> int:
        """How many components run on this install."""
        return len(self.components) - len(self.blocked_components)

    def components_of(self, kind: str) -> tuple[ComponentStatus, ...]:
        """Components of one registry kind (e.g. ``"model"``), in registry order."""
        return tuple(component for component in self.components if component.kind == kind)

    def to_markdown(self) -> str:
        """Render the report as plain markdown (environment, extras, summary)."""
        from tulip.evaluation._format import markdown_table

        env = f"tulip {self.tulip_version} | Python {self.python_version} | {self.platform}"
        extra_rows = [
            (
                extra.name,
                "yes" if extra.installed else "no",
                extra.purpose,
                "" if extra.installed else extra.install_hint,
            )
            for extra in self.extras
        ]
        summary = f"{self.runnable_count} of {len(self.components)} components runnable now"
        table = markdown_table(("extra", "installed", "unlocks", "install"), extra_rows)
        return f"# tulip doctor\n\n{env}\n\n{table}\n\n{summary}"


def probe_extras() -> tuple[ExtraStatus, ...]:
    """Probe every optional extra, in name order, without importing heavy modules."""
    statuses: list[ExtraStatus] = []
    for name, (modules, purpose) in _EXTRAS.items():
        missing = tuple(module for module in modules if not is_available(module))
        statuses.append(
            ExtraStatus(
                name=name,
                installed=not missing,
                purpose=purpose,
                modules=modules,
                missing_modules=missing,
            )
        )
    return tuple(statuses)


def component_statuses(
    extras: tuple[ExtraStatus, ...] | None = None,
) -> tuple[ComponentStatus, ...]:
    """Availability of every model, feature, and explainer on this install.

    A component is available when it declares no extra (core install) or when its
    declared extra is installed. The extra for each component is read from its
    registration metadata, so this function needs no per-component knowledge.

    Args:
        extras: Pre-probed extras to reuse; probed fresh when omitted.
    """
    from tulip.explain import EXPLAINERS
    from tulip.features import AUDIO_FEATURES, TEXT_FEATURES
    from tulip.models import MODELS

    installed = {extra.name for extra in (extras or probe_extras()) if extra.installed}
    statuses: list[ComponentStatus] = []
    for registry in (MODELS, TEXT_FEATURES, AUDIO_FEATURES, EXPLAINERS):
        for name in registry.names():
            extra = registry.metadata(name).get("extra")
            statuses.append(
                ComponentStatus(
                    kind=registry.kind,
                    name=name,
                    extra=extra,
                    available=extra is None or extra in installed,
                )
            )
    return tuple(statuses)


def run_doctor() -> DoctorReport:
    """Assemble the full diagnostic for the current interpreter and install."""
    extras = probe_extras()
    return DoctorReport(
        python_version=platform.python_version(),
        platform=f"{platform.system()} {platform.machine()}".strip(),
        tulip_version=__version__,
        extras=extras,
        components=component_statuses(extras),
    )
