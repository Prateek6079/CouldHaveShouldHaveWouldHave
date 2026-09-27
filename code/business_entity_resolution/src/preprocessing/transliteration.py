"""Deterministic Indic-to-Roman conversion; originals remain only in input TSVs."""
import re
import unicodedata as ud

from indic_transliteration import sanscript

# Unicode blocks observed in read-only samples, mapped to supported schemes.
SCRIPTS = (
    (0x0900, 0x097F, sanscript.DEVANAGARI),
    (0x0980, 0x09FF, sanscript.BENGALI),
    (0x0A00, 0x0A7F, sanscript.GURMUKHI),
    (0x0A80, 0x0AFF, sanscript.GUJARATI),
    (0x0B00, 0x0B7F, sanscript.ORIYA),
    (0x0B80, 0x0BFF, sanscript.TAMIL),
    (0x0C00, 0x0C7F, sanscript.TELUGU),
    (0x0C80, 0x0CFF, sanscript.KANNADA),
    (0x0D00, 0x0D7F, sanscript.MALAYALAM),
)
RUNS = [(re.compile(f'[{chr(lo)}-{chr(hi)}\\u200c\\u200d]+'), scheme)
        for lo, hi, scheme in SCRIPTS]
# Approximate unsupported modern characters using supported script sequences.
# Chillu letters denote consonants without an inherent vowel (Unicode ch. 12).
SCRIPT_ALIASES = str.maketrans({
    'ൺ': 'ണ്', 'ൻ': 'ന്', 'ർ': 'ര്', 'ൽ': 'ല്', 'ൾ': 'ള്', 'ൿ': 'ക്',
    'ऑ': 'ओ', 'ॉ': 'ो', 'ऍ': 'ए', 'ॅ': 'े', 'ୱ': 'ଵ',
})
LATIN_ALIASES = str.maketrans({'œ': 'oe', 'æ': 'ae', 'ø': 'o', 'ł': 'l',
                              'đ': 'd', 'ð': 'd', 'þ': 'th', 'ı': 'i'})


class UnsupportedScriptError(ValueError):
    """A letter could not be represented by the selected Romanization rules."""


def romanize(text: str) -> str:
    """NFKC cleanup, script-run transliteration to IAST, then Latin accent folding.

    Mixed Latin/Indic text is supported. This is not translation or learned
    spelling conversion. Unsupported letters fail rather than disappear.
    """
    text = ud.normalize('NFKC', text)
    if text.isascii():
        return text.casefold()
    text = ''.join(str(ud.decimal(c)) if c.isdecimal() else c for c in text)
    text = text.translate(SCRIPT_ALIASES)
    for pattern, scheme in RUNS:
        text = pattern.sub(lambda match: sanscript.transliterate(match[0], scheme, sanscript.IAST), text)
    text = text.casefold().translate(LATIN_ALIASES)
    # Fold accents only after the Indic script has been converted. Preserve a
    # conventional 'sh' spelling for IAST sibilants before removing diacritics.
    text = text.replace('ś', 'sh').replace('ṣ', 'sh')
    text = ''.join(c for c in ud.normalize('NFKD', text)
                   if not ud.category(c).startswith('M') and ud.category(c) != 'Cf')
    for c in text:
        if c.isalpha() and not ('a' <= c <= 'z'):
            raise UnsupportedScriptError(f'Unsupported letter U+{ord(c):04X} ({ud.name(c, "unknown")})')
    return text
