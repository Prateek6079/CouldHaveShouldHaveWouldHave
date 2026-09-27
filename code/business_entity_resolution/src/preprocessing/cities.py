"""Country-scoped reviewed city aliases augment, rather than replace, parsing."""
from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path
import re
import unicodedata

from .address_schema import AddressResult

STATE_CODES = set('AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY'.lower().split())


def canonical(text: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', text.casefold()) if not unicodedata.combining(c))


def key(text: str) -> str:
    text = ''.join(c for c in unicodedata.normalize('NFKD', text.casefold()) if not unicodedata.combining(c))
    return ' '.join(re.sub(r"[.'’]", '', text).split())


@lru_cache(maxsize=1)
def city_index(country="us"):
    with (Path(__file__).resolve().parents[1] / 'resources/city_aliases.json').open('r',encoding='utf-8') as handle:
        data=json.load(handle)
    if data['version'] != 1:
        raise ValueError('Unsupported city alias version')
    index={}
    for entry in data[country]:
        for alias in [entry['city'], *entry['aliases']]:
            value=(canonical(entry['city']), entry.get('state', entry.get('requires_context', '')).lower())
            if value not in index.setdefault(key(alias),[]): index[key(alias)].append(value)
    return index, {key(a) for a in data['regional_aliases']}


def enrich_us_city(result: AddressResult, text: str | None) -> AddressResult:
    if not text or result.parse_status == 'empty': return result
    index, regional=city_index()
    segments=[key(s.strip(' ,;')) for s in re.split('[,;]',text)]
    state=key(result.state or '') or None
    if state is None:
        # Only explicit isolated state abbreviations; do not infer state from city.
        states={s for s in segments if s in STATE_CODES}
        if len(states)==1: state=states.pop()

    def lookup(candidate):
        candidate=key(candidate)
        hits=index.get(candidate,[])
        # Short/airport/metro aliases need supporting state context.
        if (len(candidate.replace(' ',''))<=3 or candidate in regional) and not state:
            return None
        hits=[city for city, expected in hits if state is None or state==expected]
        return hits[0] if len(set(hits))==1 else None

    # Accept the model's city guess first. Preserve unknown or conflicting guesses.
    if result.city:
        canonical=lookup(result.city)
        if canonical and canonical!=result.city:
            return replace(result,city=canonical)
        return result

    hits=set()
    for segment in segments:
        # Allow a city segment to include a trailing state/ZIP, but never scan
        # substrings of streets or names (e.g. '123 Austin St').
        segment=re.sub(r'\s+\d{5}(?:-\d{4})?$', '',segment)
        parts=segment.split()
        if len(parts)>1 and parts[-1] in STATE_CODES:
            trailing=parts[-1]
            if state is None: state=trailing
            if state==trailing: segment=' '.join(parts[:-1])
        candidate=lookup(segment)
        if candidate: hits.add(candidate)
    if len(hits)!=1: return result
    # Keep unparsed evidence and status; finding a city does not certify the parse.
    return replace(result,city=hits.pop())


def enrich_french_city(result: AddressResult, text: str | None) -> AddressResult:
    if not text or result.parse_status == 'empty': return result
    index, _ = city_index('france')
    context = key(text)
    reunion = 'reunion' in context or bool(re.search(r'(?<!\d)974\d{2}(?!\d)', context))

    def lookup(candidate):
        hits = {city for city, requirement in index.get(key(candidate), [])
                if not requirement or requirement == 'reunion' and reunion}
        return next(iter(hits)) if len(hits) == 1 else None

    if result.city:
        match = lookup(result.city)
        return replace(result, city=match) if match else result
    hits = set()
    for segment in re.split('[,;]', text):
        candidate = key(segment.strip())
        candidate = re.sub(r'^\d{5}\s+', '', candidate)
        candidate = re.sub(r'\s+cedex(?:\s+\d+)?$', '', candidate)
        match = lookup(candidate)
        if match: hits.add(match)
    return replace(result, city=hits.pop()) if len(hits) == 1 else result
