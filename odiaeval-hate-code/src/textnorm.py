"""Text normalisation and Odia-script purity scoring.

Pure functions, no dependencies beyond the stdlib. Test-covered (spec §8):
script-purity scoring on Odia, Latin and mixed strings.
"""

from __future__ import annotations

import unicodedata

# Zero-width / BOM / soft-hyphen characters stripped during normalisation (§4④).
_ZERO_WIDTH = dict.fromkeys(
    [0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0xFEFF, 0x00AD],
    None,
)

# Odia Unicode block.
ODIA_BLOCK_START = 0x0B00
ODIA_BLOCK_END = 0x0B7F


def strip_zero_width(text: str) -> str:
    return text.translate(_ZERO_WIDTH)


def collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def normalise_for_dedup(text: str) -> str:
    """Exact-dedup normal form: NFC, zero-width stripped, whitespace collapsed.

    Order matters: strip zero-width *before* collapsing so a zero-width space
    between two words does not fuse them.
    """
    text = unicodedata.normalize("NFC", text)
    text = strip_zero_width(text)
    text = collapse_whitespace(text)
    return text


def _is_letter(ch: str) -> bool:
    """True for any Unicode letter (category ``L*``)."""
    return unicodedata.category(ch)[0] == "L"


def odia_script_purity(text: str) -> float:
    """Fraction of *alphabetic* codepoints that fall in the Odia block.

    Definition (pinned in ``config.yaml`` under ``extract`` — do not drift):

    * denominator = count of codepoints with Unicode category ``L*`` (any letter:
      Odia, Latin, Devanagari, ...).
    * numerator   = those letters whose codepoint is in ``U+0B00..U+0B7F``.
    * digits, punctuation, whitespace, symbols and combining marks are excluded
      from both, so embedded numerals / punctuation / emoji do not move the score.
    * a string with no letters at all scores ``0.0``.
    """
    letters = 0
    odia = 0
    for ch in text:
        if not _is_letter(ch):
            continue
        letters += 1
        if ODIA_BLOCK_START <= ord(ch) <= ODIA_BLOCK_END:
            odia += 1
    if letters == 0:
        return 0.0
    return odia / letters
