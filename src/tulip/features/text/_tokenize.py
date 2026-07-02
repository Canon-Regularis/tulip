"""Shared word tokenization for text feature extractors.

A single tokenizer definition keeps lexical statistics comparable across the
stylometry, affix, and keyword extractors: all three count the same "words".
Tokens are runs of Unicode letters only (no digits, no underscore), so Polish
diacritics are preserved exactly and numbers never pollute lexical features.
"""

from __future__ import annotations

import re

#: Runs of Unicode letters. ``\w`` minus digits and underscore keeps ``godać``
#: intact while excluding ``r2d2``-style tokens and numeric noise.
_WORD_RE = re.compile(r"[^\W\d_]+")


def word_tokens(text: str, *, lowercase: bool = False) -> list[str]:
    """Split ``text`` into letter-only word tokens.

    Args:
        text: Raw input text.
        lowercase: Lowercase tokens (Unicode-aware, diacritics untouched).

    Returns:
        The list of word tokens in document order (possibly empty).
    """
    tokens = _WORD_RE.findall(text)
    if lowercase:
        tokens = [token.lower() for token in tokens]
    return tokens
