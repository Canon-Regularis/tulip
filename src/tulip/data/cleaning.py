"""Composable text normalisation for dialect corpora.

:class:`TextCleaner` applies a fixed sequence of individually toggleable
steps. The guiding principle is that *dialectal orthography is signal*: the
cleaner normalises encoding, typography, and transcription artifacts, but it
never strips diacritics, never respells dialect forms, and never "corrects"
non-standard grammar -- those are exactly the features the classifiers need.

Steps (applied in this order when enabled):

1. ``nfc`` -- Unicode NFC normalisation, so byte-identical text compares
   equal regardless of how diacritics were encoded.
2. ``remove_artifacts`` -- transcription artifacts: bracketed annotations
   (``[smiech]``, ``{pauza}``, ``<laugh>``), parenthesised annotation
   keywords (``(niezrozumiale)``), stand-alone pause markers (`` ... ``,
   ``--``), and repeated-punctuation squeeze (``???`` -> ``?``).
3. ``normalise_punctuation`` -- typographic quotes/dashes/ellipsis to their
   plain ASCII equivalents, so the same utterance typed with different
   editors yields identical features.
4. ``collapse_whitespace`` -- any whitespace run to a single space, trimmed.
5. ``lowercase`` -- optional casefolding (off by default: capitalisation
   carries stylometric signal).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Iterator
from typing import Any

from tulip.core.types import Sample
from tulip.utils.logging import get_logger

_logger = get_logger(__name__)

# Typographic quotes -> ASCII. Written as escapes to keep the source
# unambiguous (and ruff RUF001-clean).
_QUOTE_MAP = {
    "„": '"',  # double low-9 quote (Polish opening)
    "“": '"',  # left double quote
    "”": '"',  # right double quote
    "‟": '"',  # double high-reversed-9 quote
    "«": '"',  # left guillemet
    "»": '"',  # right guillemet
    "‘": "'",  # left single quote
    "’": "'",  # right single quote (also apostrophe)
    "‚": "'",  # single low-9 quote
    "‛": "'",  # single high-reversed-9 quote
    "‹": "'",  # left single guillemet
    "›": "'",  # right single guillemet
}
_QUOTE_TRANSLATION = str.maketrans(_QUOTE_MAP)

# Figure dash, en dash, em dash, horizontal bar, minus sign -> hyphen-minus.
_DASH_RE = re.compile("[‒–—―−]")
_ELLIPSIS = "…"  # horizontal ellipsis as a single codepoint

# Bracketed transcription annotations are removed wholesale; square/curly/
# angle brackets are annotation markup in every corpus we ingest.
_BRACKET_ANNOTATION_RE = re.compile(r"\[[^\][]*\]|\{[^{}]*\}|<[^<>]*>")

# Parentheses also occur in ordinary prose, so only parenthesised *annotation
# keywords* are removed (Polish transcription conventions + English fallbacks).
_ANNOTATION_KEYWORDS = (
    r"[sś]miech\w*",
    r"niezrozumia\w*",
    r"pauza\w*",
    r"przerwa\w*",
    r"ha[lł]as\w*",
    r"kaszel|kas[lł]e\w*",
    r"wzdych\w*",
    r"p[lł]acz\w*",
    r"cisza",
    r"szum\w*",
    r"oklaski",
    r"[sś]piew\w*",
    r"chrz[aą]k\w*",
    r"laugh\w*",
    r"pause",
    r"noise",
    r"inaudible",
    r"unintelligible",
    r"cough\w*",
)
_PAREN_ANNOTATION_RE = re.compile(
    r"\(\s*(?:" + "|".join(_ANNOTATION_KEYWORDS) + r")[^()]*\)|\(\s*[.\-*\s]+\)",
    re.IGNORECASE,
)

# Stand-alone (whitespace-bounded) dash runs and ellipses are pause markers
# in speech transcription; attached ones ("wiesz...") are kept as punctuation.
_PAUSE_MARKER_RE = re.compile(r"(?<!\S)(?:-{1,}|\.{3,}|…)(?!\S)")

# Repeated-punctuation squeeze: runs of the same mark collapse to one; runs
# of four or more dots collapse to a plain "..." ellipsis.
_REPEAT_PUNCT_RE = re.compile(r"([!?,;:])\1+")
_MANY_DOTS_RE = re.compile(r"\.{4,}")

_WHITESPACE_RE = re.compile(r"\s+")


class TextCleaner:
    """Composable, order-fixed text normaliser for dialect text.

    Each step is an independent constructor flag so experiments can ablate
    them; ``config()`` echoes the flags for dataset manifests. The cleaner is
    stateless and safe to share across threads.

    Args:
        nfc: Apply Unicode NFC normalisation.
        remove_artifacts: Strip transcription annotations, pause markers, and
            squeeze repeated punctuation.
        normalise_punctuation: Map typographic quotes/dashes/ellipsis to
            ASCII equivalents.
        collapse_whitespace: Collapse whitespace runs to single spaces and trim.
        lowercase: Casefold the text (off by default).
    """

    def __init__(
        self,
        *,
        nfc: bool = True,
        remove_artifacts: bool = True,
        normalise_punctuation: bool = True,
        collapse_whitespace: bool = True,
        lowercase: bool = False,
    ) -> None:
        self.nfc = nfc
        self.remove_artifacts = remove_artifacts
        self.normalise_punctuation = normalise_punctuation
        self.collapse_whitespace = collapse_whitespace
        self.lowercase = lowercase

    def clean(self, text: str) -> str:
        """Return ``text`` with all enabled steps applied (in fixed order)."""
        if self.nfc:
            text = unicodedata.normalize("NFC", text)
        if self.remove_artifacts:
            text = _BRACKET_ANNOTATION_RE.sub(" ", text)
            text = _PAREN_ANNOTATION_RE.sub(" ", text)
            text = _PAUSE_MARKER_RE.sub(" ", text)
            text = _MANY_DOTS_RE.sub("...", text)
            text = _REPEAT_PUNCT_RE.sub(r"\1", text)
        if self.normalise_punctuation:
            text = text.translate(_QUOTE_TRANSLATION)
            text = _DASH_RE.sub("-", text)
            text = text.replace(_ELLIPSIS, "...")
        if self.collapse_whitespace:
            text = _WHITESPACE_RE.sub(" ", text).strip()
        if self.lowercase:
            text = text.lower()
        return text

    __call__ = clean

    def clean_sample(self, sample: Sample) -> Sample:
        """Return a copy of ``sample`` with cleaned text (audio-only pass through)."""
        if sample.text is None:
            return sample
        cleaned = self.clean(sample.text)
        if cleaned == sample.text:
            return sample
        return sample.model_copy(update={"text": cleaned})

    def clean_samples(self, samples: Iterable[Sample]) -> Iterator[Sample]:
        """Lazily clean a sample stream."""
        for sample in samples:
            yield self.clean_sample(sample)

    def config(self) -> dict[str, Any]:
        """Return the step flags, for manifests and experiment echo."""
        return {
            "nfc": self.nfc,
            "remove_artifacts": self.remove_artifacts,
            "normalise_punctuation": self.normalise_punctuation,
            "collapse_whitespace": self.collapse_whitespace,
            "lowercase": self.lowercase,
        }

    def __repr__(self) -> str:
        flags = ", ".join(f"{key}={value}" for key, value in self.config().items())
        return f"TextCleaner({flags})"


__all__ = ["TextCleaner"]
