"""Project-owned conservative rules for French address extraction."""
import re

from .address_schema import AddressResult, clean
from .cities import enrich_french_city
from .address_normalization import normalize_french_street, french_street_pattern

POSTAL = re.compile(r'(?<![\w-])\d{5}(?![\w-])')


def _rules(text: str) -> AddressResult:
    values, spans = {}, []
    codes = list(POSTAL.finditer(text))
    if len(codes) == 1:
        code = codes[0]
        values['postal_code'] = code[0]
        spans.append(code.span())
        tail = text[code.end():]
        # City extraction is restricted to letters, spaces, hyphens/apostrophes.
        # CEDEX and country trailers are left unassigned rather than made city names.
        city = re.match(r"\s*([a-z][a-z '\-]*?)(?=\s+cedex\b|[,;]|$)", tail, re.I)
        if city and city[1].strip().lower() not in ('france','fr'):
            values['city'] = clean(city[1])
            spans.append((code.end()+city.start(), code.end()+city.end()))
    prefix_end = codes[0].start() if len(codes) == 1 else len(text)
    street = french_street_pattern().match(text[:prefix_end])
    if street:
        values['house_number'] = clean(street['number'])
        values['street'] = normalize_french_street(street['street'])
        spans.append(street.span())
    leftover = list(text)
    for start, end in spans:
        leftover[start:end] = ' ' * (end-start)
    unparsed = clean(''.join(leftover))
    # Rules always report partial: extraction is not postal validation.
    return AddressResult(**values, unparsed_text=unparsed, parser='fr_rules',
                         parse_status='partial' if values else 'fallback', parse_reason='conservative_rules')


def _parse_address(text: str | None) -> AddressResult:
    if not text or not text.strip():
        return AddressResult(parser='fr_rules', parse_status='empty')
    return _rules(text)


def parse_address(text: str | None) -> AddressResult:
    return enrich_french_city(_parse_address(text), text)
