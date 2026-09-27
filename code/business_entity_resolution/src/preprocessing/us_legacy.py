"""Retained usaddress baseline for explicit comparison; not the default parser."""
import re

import usaddress

from .address_schema import AddressResult, clean
from .cities import STATE_CODES, enrich_us_city
from .address_normalization import normalize_component, rules

LABELS = {
    'house_number': ('AddressNumberPrefix', 'AddressNumber', 'AddressNumberSuffix'),
    'street': ('StreetNamePreModifier', 'StreetNamePreDirectional', 'StreetNamePreType',
               'StreetName', 'StreetNamePostType', 'StreetNamePostDirectional', 'StreetNamePostModifier'),
    'unit': ('OccupancyType', 'OccupancyIdentifier'),
    'floor': ('SubaddressType', 'SubaddressIdentifier'),
    'building': ('BuildingName',),
    'city': ('PlaceName',), 'state': ('StateName',), 'postal_code': ('ZipCode',),
    'po_box': ('USPSBoxType', 'USPSBoxID', 'USPSBoxGroupType', 'USPSBoxGroupID'),
}


def _aliases() -> dict:
    """Expose US aliases for the retention audit's equivalence checks."""
    return rules()['us']


def _state_code_from_text(text: str | None) -> str | None:
    """Recover a unique postal code from a complete segment or segment suffix."""
    if not text:
        return None
    candidates = set()
    for segment in re.split(r'[,;]', text.lower()):
        segment = re.sub(r"[^a-z0-9\s]", " ", segment).strip()
        words = segment.split()
        if segment in STATE_CODES:
            candidates.add(segment)
        elif words and words[-1] in STATE_CODES:
            candidates.add(words[-1])
    return next(iter(candidates)) if len(candidates) == 1 else None


def _remove_state_code(text: str | None, state: str) -> str | None:
    """Remove only the recognized code as a segment or final segment token."""
    if not text:
        return text
    remaining = []
    for segment in re.split(r"[,;]", text):
        words = segment.split()
        normalized = re.sub(r"[^a-z0-9\s]", " ", segment.lower()).strip()
        if normalized == state:
            continue
        if len(words) > 1 and words[-1].lower().strip(".,;") == state:
            segment = " ".join(words[:-1])
        if segment.strip():
            remaining.append(segment.strip())
    return clean(", ".join(remaining))


def _parse_address(text: str | None) -> AddressResult:
    if not text or not text.strip():
        return AddressResult(parser='usaddress', parse_status='empty')
    try:
        tags, kind = usaddress.tag(text)
    except (usaddress.RepeatedLabelError, ValueError) as exc:
        return AddressResult(parser='usaddress', parse_status='fallback',
                             parse_reason=type(exc).__name__, unparsed_text=text)
    values, used = {}, set()
    for field, labels in LABELS.items():
        parts = []
        for label in labels:
            if label not in tags:
                continue
            value = clean(tags[label])
            if value is None:
                continue
            # SubaddressType also includes non-floor information; do not call it a floor.
            if field == 'floor' and tags.get('SubaddressType', '').lower().rstrip('.') not in ('floor', 'fl', 'flr'):
                continue
            if label in ('StreetNamePreDirectional', 'StreetNamePostDirectional'):
                value = normalize_component(value, 'us', 'directions')
            if label in ('StreetNamePreType', 'StreetNamePostType'):
                value = normalize_component(value, 'us', 'street_types')
            parts.append(value)
            used.add(label)
        if parts:
            values[field] = ' '.join(parts)
    # Model guesses that violate basic numeric structure remain unassigned.
    postal = values.get('postal_code')
    if postal and not re.fullmatch(r'\d{5}(?:-\d{4})?', postal):
        values.pop('postal_code')
        used.discard('ZipCode')
    values['unparsed_text'] = clean(' '.join(value for key, value in tags.items() if key not in used))
    if not values.get('state') and 'Recipient' in tags:
        state_code = _state_code_from_text(text)
        if state_code:
            values['state'] = state_code
            values['unparsed_text'] = _remove_state_code(values['unparsed_text'], state_code)
    complete = (values.get('street') and values.get('house_number') or values.get('po_box')) and values.get('city') and values.get('state')
    status = 'parsed' if complete and not values['unparsed_text'] else 'partial'
    if not any(values.get(field) for field in LABELS):
        status = 'fallback'
        values['unparsed_text'] = text
    return AddressResult(**values, parser='usaddress', parse_status=status,
                         parse_reason=kind if status != 'parsed' else None)


def parse_address(text: str | None) -> AddressResult:
    return enrich_us_city(_parse_address(text), text)
