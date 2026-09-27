"""Compact normalized text expressions; address separators retain meaning."""
import polars as pl

from .transliteration import romanize

PUNCTUATION = str.maketrans({'’': "'", '‘': "'", 'ʼ': "'", '‐': '-', '‑': '-',
                            '–': '-', '—': '-', '−': '-', '⁄': '/', '／': '/'})


def romanize_series(series: pl.Series) -> pl.Series:
    """Convert only distinct non-ASCII values in the current batch using Python.

    ASCII records avoid per-record Python callbacks. The replacement map lives
    for one batch only. Null names/addresses become empty strings.
    """
    clean = series.fill_null('').str.normalize('NFKC')
    unusual = clean.filter(clean.str.contains(r'[^\x00-\x7f]')).unique()
    if len(unusual):
        converted = [romanize(value.translate(PUNCTUATION)) for value in unusual]
        clean = clean.replace_strict(unusual, pl.Series(converted, dtype=pl.String), default=clean)
    return clean


def normalized_expr(column: str, *, address: bool = False) -> pl.Expr:
    """Lowercase and clean Roman text without expanding address abbreviations."""
    value = pl.col(column).str.to_lowercase()
    if address:
        # Preserve unit numbers, numeric ranges, street boundaries and markers.
        value = value.str.replace_all(r"[^a-z0-9\s/.,#'&()+-]", ' ')
    else:
        value = (value.str.replace_all('&', ' and ')
                 .str.replace_all("'", '')
                 .str.replace_all(r'[^a-z0-9\s]', ' '))
    return value.str.replace_all(r'\s+', ' ').str.strip_chars()
