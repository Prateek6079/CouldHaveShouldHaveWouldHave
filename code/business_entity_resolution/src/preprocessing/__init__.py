"""Read-only batch preprocessing: one Roman name/address representation."""
from collections.abc import Iterable, Iterator

import polars as pl

from .business_name import business_name_expr
from .address_normalization import normalize_india_address
from .text import normalized_expr, romanize_series
from .transliteration import UnsupportedScriptError


def preprocess_batch(batch: pl.DataFrame) -> pl.DataFrame:
    """Return a new frame replacing name/address text; preserve IDs and country.

    No originals, transliteration copies or legal-form columns are appended.
    Input frames and source files are not mutated. Address parsing is a later
    stage; punctuation carrying unit/range information is preserved here.
    """
    required = {'entity_id', 'business_name', 'business_address', 'country'}
    if missing := required - set(batch.columns):
        raise ValueError(f'Missing preprocessing columns: {sorted(missing)}')
    for column in required:
        if batch.schema[column] not in (pl.String, pl.Null):
            raise ValueError(f'{column} must be a string column')
    replacements = []
    for column in ('business_name', 'business_address'):
        try:
            replacements.append(romanize_series(batch[column].cast(pl.String)).alias(column))
        except UnsupportedScriptError as exc:
            raise UnsupportedScriptError(f'{column}: {exc}') from exc
    result = batch.with_columns(replacements)
    result = result.with_columns(normalized_expr('business_name'),
                                 normalized_expr('business_address', address=True))
    # Country is unchanged in output; null-only input needs a string expression.
    if result.schema['country'] == pl.Null:
        result = result.with_columns(pl.col('country').cast(pl.String))
    india = pl.col('country').str.to_lowercase().str.strip_chars().is_in(['india', 'in', 'ind'])
    # Normalize each distinct India address once per batch. Besides Roman/PIN
    # forms this separates high-confidence concatenated structural markers.
    addresses = result['business_address']
    candidates = result.filter(india)['business_address'].unique()
    if len(candidates):
        normalized = addresses.replace_strict(
            candidates, pl.Series([normalize_india_address(v) for v in candidates]), default=addresses)
        result = result.with_columns(pl.when(india).then(normalized).otherwise(pl.col('business_address')).alias('business_address'))
    return result.with_columns(business_name_expr())


def preprocess_batches(batches: Iterable[pl.DataFrame]) -> Iterator[pl.DataFrame]:
    """Process one batch at a time, without accumulating the complete dataset."""
    for batch in batches:
        yield preprocess_batch(batch)
