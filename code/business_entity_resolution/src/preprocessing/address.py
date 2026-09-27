"""Country routing and bounded-cache Polars enrichment."""
from collections.abc import Iterable, Iterator
from functools import lru_cache

import polars as pl

from . import france, generic, india, us
from .address_schema import ADDRESS_COLUMNS, AddressResult

US = {'us','usa','united states','united states of america','u.s.','u.s.a.'}
FRANCE = {'fr','fra','france'}
INDIA = {'in','ind','india'}


def parse_address(text: str | None, country: str | None) -> AddressResult:
    country = ' '.join((country or '').lower().split())
    if country in US:
        return us.parse_address(text)
    if country in FRANCE:
        return france.parse_address(text)
    if country in INDIA:
        return india.parse_address(text)
    return generic.parse_address(text, reason='unsupported_country')


def enrich_address_batch(batch: pl.DataFrame) -> pl.DataFrame:
    """Append nullable address_* fields to an already text-preprocessed batch.

    Whole-address text stays unchanged. Parse-status is extraction completeness,
    not confidence or deliverability. A cache of at most 4096 addresses is local
    to this call. Existing component columns are replaced on repeated enrichment.
    """
    for column in ('business_address','country'):
        if column not in batch.columns or batch.schema[column] not in (pl.String,pl.Null):
            raise ValueError(f'{column} must be a string column')

    @lru_cache(maxsize=4096)
    def cached(text, country):
        return parse_address(text,country).values()

    rows = [cached(text,country) for text,country in batch.select('business_address','country').iter_rows()]
    components = pl.DataFrame(rows, schema={name:pl.String for name in ADDRESS_COLUMNS}, orient='row')
    return batch.with_columns(components)


def enrich_address_batches(batches: Iterable[pl.DataFrame]) -> Iterator[pl.DataFrame]:
    for batch in batches:
        yield enrich_address_batch(batch)
