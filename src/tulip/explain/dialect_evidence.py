"""Dialectological evidence explanation: attribute a prediction to named phenomena.

Registers ``dialect_evidence`` in the :data:`tulip.explain.registry.EXPLAINERS`
registry.

Where ``top_tfidf`` reports opaque coefficient-weighted features
(``char: sci``), this explainer reports the linguistic evidence a dialectologist
would cite: *which* marker lexemes matched (``baca`` -> Podhale) and *which*
isoglosses fired (``psiwo`` -> soft labials -> Masovian). It makes a fine-grained
dialect call justifiable in domain terms, and produces a serialisable
:class:`~tulip.core.types.Explanation` that model cards, error analysis, and the
demo UI can render.

It is pure composition over the shared resources -- it re-implements no
detection. Lexical evidence comes from the marker lexicon
(:func:`tulip.features.text.keywords.load_lexicon`, mapped to families by
:func:`~tulip.features.text.keywords.family_for_lexicon_key`); phonological
evidence comes from the rewrite rules
(:func:`tulip.features.text.phonological_rules.load_phonological_rules`), using
their own ``fired_matches`` / ``applicable_matches``.

Honesty caveat (carried in the explanation ``details``, mirroring the attention
explainer): this evidence is *resource-defined* -- it is what the lexicon and
isogloss rules find in the text, not a proof of what the model actually keyed on.
The model's own prediction is reported alongside so the two can be compared.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tulip.core.types import Explanation, TokenAttribution
from tulip.explain._shared import as_text
from tulip.explain.registry import EXPLAINERS
from tulip.features.text._tokenize import word_tokens
from tulip.features.text.keywords import family_for_lexicon_key, load_lexicon
from tulip.features.text.phonological_rules import load_phonological_rules
from tulip.utils.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from tulip.features.text.phonological_rules import PhonologicalRule

__all__ = ["DialectEvidenceExplainer"]

logger = get_logger(__name__)

#: Fixed diagnosticity weights, so an :class:`Explanation` ranks a fired sound
#: change above a lexeme (a lexeme can be borrowed; a fired isogloss is stronger
#: positive evidence) and both above the mere presence of a merger's standard
#: environment (which is evidence *against* the change having fired).
_FIRED_WEIGHT = 2.0
_MARKER_WEIGHT = 1.5
_APPLICABLE_WEIGHT = -0.5

_CAVEAT = (
    "Evidence is resource-defined (marker lexicon + isogloss rules), not a proof "
    "of what the model attended to; the model's own prediction is reported for comparison."
)


@EXPLAINERS.register("dialect_evidence")
class DialectEvidenceExplainer:
    """Attribute a text's prediction to named dialectal phenomena.

    Args:
        lexicon_path: YAML marker lexicon replacing the bundled one.
        rules_path: YAML rule file replacing the bundled one.
        top_k: Maximum number of attributions to return.
    """

    def __init__(
        self,
        lexicon_path: str | Path | None = None,
        rules_path: str | Path | None = None,
        top_k: int = 15,
    ) -> None:
        self.lexicon_path = lexicon_path
        self.rules_path = rules_path
        self.top_k = top_k

    def explain(self, pipeline: Any, raw_input: Any, **kwargs: Any) -> Explanation:
        """Explain one prediction as the dialectal evidence present in the text.

        Args:
            pipeline: A fitted classifier exposing ``predict`` (used only to
                report the model's own prediction alongside the evidence).
            raw_input: The raw text to explain.
            **kwargs: ``top_k`` overrides the constructor value.

        Returns:
            An :class:`Explanation` whose ``attributions`` name the matched
            markers and fired isoglosses (strongest evidence first) and whose
            ``details`` carry the structured per-phenomenon breakdown, the
            model's predicted label, and the resource-defined caveat.
        """
        top_k = int(kwargs.get("top_k", self.top_k))
        text = as_text(raw_input)
        lexicon = load_lexicon(self.lexicon_path)
        rules = load_phonological_rules(self.rules_path)
        tokens = word_tokens(text, lowercase=True)

        markers = _marker_evidence(tokens, lexicon)
        fired, applicable = _rule_evidence(tokens, rules)

        attributions = _attributions(markers, fired, applicable)
        ordered = sorted(attributions, key=lambda a: abs(a.weight), reverse=True)[:top_k]

        return Explanation(
            method="dialect_evidence",
            predicted_label=_predicted_label(pipeline, text),
            attributions=tuple(ordered),
            details={
                "markers": [m.as_detail() for m in markers],
                "fired_rules": [f.as_detail() for f in fired],
                "applicable_rules": [a.as_detail() for a in applicable],
                "families": _family_tally(markers, fired),
                "caveat": _CAVEAT,
            },
        )


class _Evidence:
    """One piece of matched evidence, tallied by surface form."""

    __slots__ = ("count", "families", "label", "phenomenon", "surface")

    def __init__(
        self, phenomenon: str, label: str, surface: str, families: tuple[str, ...]
    ) -> None:
        self.phenomenon = phenomenon
        self.label = label
        self.surface = surface
        self.families = families
        self.count = 1

    def as_detail(self) -> dict[str, Any]:
        return {
            "phenomenon": self.phenomenon,
            "label": self.label,
            "surface": self.surface,
            "families": list(self.families),
            "count": self.count,
        }


def _marker_evidence(tokens: list[str], lexicon: dict[str, tuple[str, ...]]) -> list[_Evidence]:
    """Lexeme markers found in the tokens, tallied by (dialect key, surface)."""
    marker_to_keys: dict[str, list[str]] = {}
    for key, markers in lexicon.items():
        for marker in markers:
            marker_to_keys.setdefault(marker, []).append(key)

    found: dict[tuple[str, str], _Evidence] = {}
    for token in tokens:
        for key in marker_to_keys.get(token, ()):
            family = family_for_lexicon_key(key)
            evidence = found.get((key, token))
            if evidence is None:
                found[(key, token)] = _Evidence(
                    phenomenon="marker",
                    label=key,
                    surface=token,
                    families=(family,) if family is not None else (),
                )
            else:
                evidence.count += 1
    return sorted(found.values(), key=lambda e: (-e.count, e.label, e.surface))


def _rule_evidence(
    tokens: list[str], rules: tuple[PhonologicalRule, ...]
) -> tuple[list[_Evidence], list[_Evidence]]:
    """Fired reflexes and (for mergers) present standard environments."""
    fired: dict[tuple[str, str], _Evidence] = {}
    applicable: dict[str, _Evidence] = {}
    for rule in rules:
        for token in tokens:
            for surface in rule.fired_matches(token):
                key = (rule.name, surface)
                if key in fired:
                    fired[key].count += 1
                else:
                    fired[key] = _Evidence("isogloss_fired", rule.name, surface, rule.families)
            # Only mergers (no detectable reflex) contribute their environment as
            # evidence; a detectable rule's environment is already covered by its
            # fired matches.
            if not rule.detectable:
                for _ in rule.applicable_matches(token):
                    if rule.name in applicable:
                        applicable[rule.name].count += 1
                    else:
                        applicable[rule.name] = _Evidence(
                            "isogloss_environment", rule.name, "", rule.families
                        )
    fired_list = sorted(fired.values(), key=lambda e: (-e.count, e.label, e.surface))
    applicable_list = sorted(applicable.values(), key=lambda e: (-e.count, e.label))
    return fired_list, applicable_list


def _attributions(
    markers: list[_Evidence], fired: list[_Evidence], applicable: list[_Evidence]
) -> list[TokenAttribution]:
    """Turn tallied evidence into signed, weighted attributions."""
    attributions: list[TokenAttribution] = []
    for evidence in markers:
        attributions.append(
            TokenAttribution(
                token=f"{evidence.label}: {evidence.surface}",
                weight=_MARKER_WEIGHT * evidence.count,
            )
        )
    for evidence in fired:
        attributions.append(
            TokenAttribution(
                token=f"{evidence.label}: {evidence.surface}", weight=_FIRED_WEIGHT * evidence.count
            )
        )
    for evidence in applicable:
        attributions.append(
            TokenAttribution(
                token=f"{evidence.label} (env)", weight=_APPLICABLE_WEIGHT * evidence.count
            )
        )
    return attributions


def _family_tally(markers: list[_Evidence], fired: list[_Evidence]) -> dict[str, int]:
    """Count positive dialectal evidence per family, sorted by family name."""
    tally: dict[str, int] = {}
    for evidence in (*markers, *fired):
        for family in evidence.families:
            tally[family] = tally.get(family, 0) + evidence.count
    return dict(sorted(tally.items()))


def _predicted_label(pipeline: Any, text: str) -> str | None:
    """The model's own predicted label, or ``None`` if the pipeline cannot predict."""
    if not hasattr(pipeline, "predict"):
        return None
    try:
        return str(pipeline.predict([text])[0])
    except Exception:
        # The evidence stands even if prediction fails; report it without a label.
        logger.debug("dialect_evidence: pipeline.predict failed; omitting predicted label")
        return None
