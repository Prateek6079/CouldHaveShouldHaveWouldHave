"""Common nullable address components and explicit parser outcomes."""
from dataclasses import dataclass, fields


@dataclass(frozen=True, slots=True)
class AddressResult:
    house_number: str | None = None
    street: str | None = None
    unit: str | None = None
    floor: str | None = None
    building: str | None = None
    locality: str | None = None
    city: str | None = None
    state: str | None = None
    postal_code: str | None = None
    po_box: str | None = None
    landmark_direction: str | None = None
    unparsed_text: str | None = None
    parser: str = 'generic'
    parse_status: str = 'fallback'
    parse_reason: str | None = None

    def values(self) -> tuple:
        return tuple(getattr(self, name) for name in FIELD_NAMES)


FIELD_NAMES = tuple(field.name for field in fields(AddressResult))
ADDRESS_COLUMNS = tuple('address_' + name for name in FIELD_NAMES)


def clean(value: str) -> str | None:
    return ' '.join(value.strip(' ,;').lower().split()) or None
