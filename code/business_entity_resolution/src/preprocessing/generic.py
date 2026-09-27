"""Fallback without country-specific guesses; never discard address text."""
from .address_schema import AddressResult


def parse_address(text: str | None, *, reason: str = 'unsupported_country') -> AddressResult:
    if not text or not text.strip():
        return AddressResult(parse_status='empty')
    return AddressResult(unparsed_text=text, parse_reason=reason)
