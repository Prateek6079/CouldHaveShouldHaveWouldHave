"""Normalize guessed component roles; do not reinterpret parser decisions."""
from functools import lru_cache
import json
from pathlib import Path
import re

from .address_schema import clean


@lru_cache(maxsize=1)
def rules() -> dict:
    with (Path(__file__).resolve().parents[1] / 'resources/address_aliases.json').open('r', encoding='utf-8') as handle:
        data = json.load(handle)
    if data.get('version') != 1:
        raise ValueError('Unsupported address alias schema')
    return data


def normalize_component(value: str, country: str, role: str) -> str:
    value = clean(value) or ''
    return rules().get(country, {}).get(role, {}).get(value.rstrip('.'), value)


def normalize_french_street(value: str) -> str:
    # A leading recognized marker in a road component is our best street-type
    # guess. Keep the rest of the street name untouched, including Saint/St.
    value = clean(value) or ''
    return re.sub(r'^([a-z]+)\.?(?=\s|$)',
                  lambda m: normalize_component(m[0], 'france', 'street_types'), value)


@lru_cache(maxsize=1)
def french_street_pattern() -> re.Pattern:
    markers = '|'.join(re.escape(x) for x in sorted(rules()['france']['street_types'], key=len, reverse=True))
    return re.compile(
        r'^\s*(?P<number>\d+(?:\s*(?:bis|ter|quater)|[a-z])?(?:[-/]\d+[a-z]?)?)\s+'
        rf'(?P<street>(?:{markers})\.?(?=\s)[^,;]*)', re.I)


# Interpret Roman numbers only next to structural India address markers.
# In particular, initials in names such as "S K C Buildings" remain unchanged.
_ROMAN = r"(?=[ivxlcdm]+\b)m{0,3}(?:cm|cd|d?c{0,3})(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})"
_ROMAN_MARKER = r"(?:floor|flr|fl|level|lvl|storey|story|phase|sector)\.?"
_ROMAN_BEFORE = re.compile(rf"\b({_ROMAN})\s+({_ROMAN_MARKER})(?=\s|[,;.)]|$)", re.I)
_ROMAN_AFTER = re.compile(rf"\b({_ROMAN_MARKER})\s+({_ROMAN})(?=\s|[,;.)]|$)", re.I)


def _roman_digits(value: str) -> str:
    values = {'i': 1, 'v': 5, 'x': 10, 'l': 50, 'c': 100, 'd': 500, 'm': 1000}
    numbers = [values[c] for c in value.lower()]
    return str(sum(-n if i + 1 < len(numbers) and n < numbers[i + 1] else n
                   for i, n in enumerate(numbers)))


@lru_cache(maxsize=1)
def _india_pin_patterns():
    path = Path(__file__).resolve().parents[1] / 'resources/india_address_lexicon.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    names = set(data['city_aliases']) | set(data['states']) | set(data['postal_markers']) | {'india'}
    location = '|'.join(re.escape(name) for name in sorted(names, key=len, reverse=True))
    # A known location/postal label is required: 155-156 is usually a range.
    attached = re.compile(rf"\b({location})([1-9]\d{{5}})\b", re.I)
    split = re.compile(rf"\b({location})[ ,:-]+([1-9]\d{{2}}) (\d{{3}})\b", re.I)
    return attached, split


@lru_cache(maxsize=1)
def _india_concatenation_patterns():
    """Compile conservative patterns for glued structural address markers."""
    number_markers = re.compile(
        r'(?<=[a-z]{3})(?=(?:plot|flat|door|survey|room|shop|office)no(?:\d|\b))', re.I)
    marker_number = re.compile(
        r'\b(rd|road|street|lane|floor|flr|level|plot|flat|door|house|survey|room|shop|sector|phase|nh|sh)'
        r'(?=(?:no)?[-/]?\d)', re.I)
    ordinal_marker = re.compile(
        r'\b(\d+(?:st|nd|rd|th))(?=(?:floor|flr|level|phase|sector)\b)', re.I)
    # Short/common suffixes such as "st", "no", and "house" cause too many
    # false splits, so only distinctive address words are accepted here.
    word_suffix = re.compile(
        r'\b([a-z]{3,})(buildings?|apartments?|complex|road|marg|floor|highway|salai|street|lane)\b', re.I)
    landmark = re.compile(r'\b([a-z]{3,})(behind|opposite)([a-z]{2,})\b', re.I)
    return number_markers, marker_number, ordinal_marker, word_suffix, landmark


def _separate_india_address_markers(value: str) -> str:
    """Expose parser-known markers; this is not general word segmentation."""
    number_markers, marker_number, ordinal_marker, word_suffix, landmark = (
        _india_concatenation_patterns())
    value = number_markers.sub(' ', value)
    value = re.sub(
        r'\b(plot|flat|door|survey|room|shop|office)no(?=\d|\b)',
        r'\1 no ', value, flags=re.I)
    value = marker_number.sub(r'\1 ', value)
    value = ordinal_marker.sub(r'\1 ', value)
    value = word_suffix.sub(r'\1 \2', value)
    value = landmark.sub(r'\1 \2 \3', value)
    return re.sub(r'\s+', ' ', value).strip()


def normalize_india_address(value: str) -> str:
    """Normalize high-confidence India address forms without general word breaking."""
    value = _separate_india_address_markers(value)
    value = _ROMAN_BEFORE.sub(lambda m: _roman_digits(m[1]) + ' ' + m[2], value)
    value = _ROMAN_AFTER.sub(lambda m: m[1] + ' ' + _roman_digits(m[2]), value)
    attached, split = _india_pin_patterns()
    value = attached.sub(lambda m: m[1] + ' ' + m[2], value)
    return split.sub(lambda m: m[1] + ' ' + m[2] + m[3], value)
