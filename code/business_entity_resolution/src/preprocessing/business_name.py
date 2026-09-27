"""Remove dictionary-defined legal forms at name boundaries only."""
from functools import lru_cache
import json
from pathlib import Path
import re

import polars as pl

RESOURCE = Path(__file__).resolve().parents[1] / 'resources/legal_suffixes.json'


@lru_cache(maxsize=1)
def legal_rules() -> dict:
    with RESOURCE.open('r', encoding='utf-8') as handle:
        rules = json.load(handle)
    if rules.get('version') != 1:
        raise ValueError('Unsupported legal_suffixes.json version')
    return rules


def _strip(value: pl.Expr, forms: list[str]) -> pl.Expr:
    alternatives = '|'.join(re.escape(x) for x in sorted(forms, key=len, reverse=True))
    # Repetition handles stacked suffixes without deleting interior words.
    start = rf'^(?:(?:{alternatives})\s+)+'
    end = rf'(?:\s+(?:{alternatives}))+$'
    return value.str.replace(start, '').str.replace(end, '').str.strip_chars()


def business_name_expr() -> pl.Expr:
    rules = legal_rules()
    original = pl.col('business_name')
    value = _strip(original, rules['common'])
    country = pl.col('country').fill_null('').str.to_lowercase().str.strip_chars()
    for group in rules['countries'].values():
        # Combine lists so alternating forms (e.g. SARL ... Ltd SAS) all strip.
        local = _strip(original, rules['common'] + group['forms'])
        value = pl.when(country.is_in(group['labels'])).then(local).otherwise(value)
    # Never turn a name consisting only of legal forms into a missing name.
    return pl.when(value == '').then(original).otherwise(value).alias('business_name')
