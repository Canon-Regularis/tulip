"""Seeded text perturbations for robustness sweeps.

Four stressors move an input along a controlled axis. Two are grounded in the
Polish phonological rules, so they are real moves along the standard-to-dialect
axis rather than generic noise:

* ``dialect_intensity_dial`` rewrites a seeded fraction of tokens from standard
  towards dialectal, using the forward rules. At level 1 it equals
  :func:`~tulip.features.text.phonological_rules.apply_rules`.
* ``standardize`` rewrites a seeded fraction back towards standard, using the
  reverse of the detectable rules.

Two are channel stressors on the raw surface, which preserve case and
punctuation:

* ``asr_noise`` drops or confuses Polish diacritics, mimicking a transcription
  channel.
* ``typo_noise`` substitutes keyboard-adjacent letters.

Every perturbation is deterministic given its ``rng``. Level 0 is identity for
all of them, so a sweep's level-0 cell equals the clean baseline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from tulip.robustness.registry import PERTURBATIONS

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    import numpy as np

    from tulip.features.text.phonological_rules import PhonologicalRule

__all__ = [
    "AsrNoise",
    "DialectIntensityDial",
    "Perturbation",
    "Standardize",
    "TypoNoise",
]


@runtime_checkable
class Perturbation(Protocol):
    """A seeded transform from one text to a perturbed text at a given level."""

    def perturb(self, text: str, *, level: float, rng: np.random.Generator) -> str:
        """Return ``text`` perturbed at ``level`` in ``[0, 1]`` (0 is identity)."""


def _rewrite_fraction(
    text: str,
    rules: tuple[PhonologicalRule, ...],
    level: float,
    rng: np.random.Generator,
    *,
    reverse: bool,
) -> str:
    """Rewrite a seeded fraction of tokens through the rules, forward or reverse."""
    from tulip.features.text._tokenize import word_tokens

    rewritten = []
    for token in word_tokens(text, lowercase=True):
        if rng.random() < level:
            for rule in rules:
                token = rule.normalize_token(token) if reverse else rule.apply_token(token)
        rewritten.append(token)
    return " ".join(rewritten)


def _substitute(
    text: str, level: float, rng: np.random.Generator, mapping: Mapping[str, Sequence[str]]
) -> str:
    """Replace each mapped character with a seeded variant with probability ``level``.

    Shared by the two channel stressors: a character whose lowercase form is in
    ``mapping`` is replaced, with probability ``level``, by a seeded choice among
    its variants; case is preserved and unmapped characters pass through. A ``str``
    value counts as a sequence of single-character variants, so both the diacritic
    map (tuples) and the keyboard map (strings) fit unchanged.
    """
    out = []
    for char in text:
        variants = mapping.get(char.lower())
        if variants and rng.random() < level:
            choice = variants[int(rng.integers(len(variants)))]
            # Title-case a capitalised source: a multi-character variant (e.g. "rz")
            # becomes "Rz", not "RZ"; for a single character this is just its upper form.
            out.append((choice[:1].upper() + choice[1:]) if char.isupper() else choice)
        else:
            out.append(char)
    return "".join(out)


class _RuleRewriter:
    """Base for the rule-grounded perturbations: load the rules, rewrite a fraction.

    Both directions load the phonological rules identically and rewrite a seeded
    fraction of tokens through them; a subclass sets :attr:`_reverse` to choose
    the direction (forward towards dialectal, or reverse towards standard). Level 0
    is identity.
    """

    _reverse: bool = False

    def __init__(self, rules_path: str | Path | None = None) -> None:
        from tulip.features.text.phonological_rules import load_phonological_rules

        self._rules = load_phonological_rules(rules_path)

    def perturb(self, text: str, *, level: float, rng: np.random.Generator) -> str:
        if level <= 0.0:
            return text
        return _rewrite_fraction(text, self._rules, level, rng, reverse=self._reverse)


@PERTURBATIONS.register("dialect_intensity_dial")
class DialectIntensityDial(_RuleRewriter):
    """Rewrite a seeded fraction of tokens standard towards dialectal.

    At ``level`` p each token is rewritten with probability p through every
    forward rule. Level 0 is identity; level 1 rewrites every token, matching
    :func:`~tulip.features.text.phonological_rules.apply_rules`. The rewrite
    lowercases and drops punctuation, the same canonicalisation the rules use.
    """


@PERTURBATIONS.register("standardize")
class Standardize(_RuleRewriter):
    """Rewrite a seeded fraction of tokens dialectal towards standard.

    The reverse of :class:`DialectIntensityDial`, using the reverse of the
    detectable rules; mergers are left untouched because they cannot be
    reversed. At level 1 it approaches
    :func:`~tulip.features.text.phonological_rules.normalize_to_standard`.
    """

    _reverse = True


#: Polish diacritics mapped to the plainer letters a transcription channel emits.
_ASR_VARIANTS: dict[str, tuple[str, ...]] = {
    "ą": ("a", "om"),
    "ć": ("c",),
    "ę": ("e", "em"),
    "ł": ("l", "w"),
    "ń": ("n",),
    "ó": ("u", "o"),
    "ś": ("s",),
    "ź": ("z",),
    "ż": ("z", "rz"),
}


@PERTURBATIONS.register("asr_noise")
class AsrNoise:
    """Drop or confuse Polish diacritics, mimicking a transcription channel.

    Each diacritic is replaced with probability ``level`` by a seeded choice
    among its plainer variants. Case is preserved; punctuation is untouched.
    """

    def perturb(self, text: str, *, level: float, rng: np.random.Generator) -> str:
        if level <= 0.0:
            return text
        return _substitute(text, level, rng, _ASR_VARIANTS)


#: Lowercase letters mapped to their QWERTY keyboard neighbours.
_KEY_NEIGHBOURS: dict[str, str] = {
    "q": "wa",
    "w": "qeas",
    "e": "wrsd",
    "r": "etdf",
    "t": "ryfg",
    "y": "tugh",
    "u": "yihj",
    "i": "uojk",
    "o": "ipkl",
    "p": "ol",
    "a": "qwsz",
    "s": "awedxz",
    "d": "serfcx",
    "f": "drtgvc",
    "g": "ftyhbv",
    "h": "gyujnb",
    "j": "huikmn",
    "k": "jiolm",
    "l": "kop",
    "z": "asx",
    "x": "zsdc",
    "c": "xdfv",
    "v": "cfgb",
    "b": "vghn",
    "n": "bhjm",
    "m": "njk",
}


@PERTURBATIONS.register("typo_noise")
class TypoNoise:
    """Substitute keyboard-adjacent letters, mimicking typing slips.

    Each letter with a neighbour set is replaced with probability ``level`` by a
    seeded adjacent key. Case is preserved; non-letters are untouched.
    """

    def perturb(self, text: str, *, level: float, rng: np.random.Generator) -> str:
        if level <= 0.0:
            return text
        return _substitute(text, level, rng, _KEY_NEIGHBOURS)
