"""Reproducible read-only India production-path coverage audit.

Run from this directory with .venv/bin/python tests/audit_india_pipeline_60000.py.
The sample is the lowest seeded entity-ID hashes per source, filtered to India.
This reports parser coverage and retention, not component accuracy.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import polars as pl
from src.config import load_config
from src.preprocessing import preprocess_batch
from src.preprocessing.address import enrich_address_batch, INDIA
from src.preprocessing.address_schema import FIELD_NAMES

SEED = 42
PER_SOURCE = 10_000
COMPONENTS = FIELD_NAMES[:FIELD_NAMES.index('unparsed_text')]
WORD_NUMBER = re.compile(r'[a-z]+|\d+', re.I)
NUMBER_GROUP = re.compile(r'\d+(?:[-/]\d+)*')
PLACEHOLDER = re.compile(r'\b(?:n/?a|none|null|nan|unknown|not available|not applicable)\b', re.I)
RURAL_ROAD = re.compile(r'\b(?:nh|sh|national highway|state highway|highway|bypass|gram|gaon|village|gali|marg|sadak|road|rd|lane|cross road)\b', re.I)
UNIT_BUILDING = re.compile(r'\b(?:flat|apt|apartment|unit|suite|shop|office|floor|flr|tower|wing|block|building|bldg|complex|bhavan|bhawan)\b|#', re.I)


def tokens(value):
    return Counter(WORD_NUMBER.findall(value.lower()))


def numbers(value):
    return Counter(NUMBER_GROUP.findall(value))


def snapshot(path):
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    return {'sha256': digest, 'mtime_ns': path.stat().st_mtime_ns}


def categorize(address, remainder, reason):
    if PLACEHOLDER.search(remainder):
        return 'placeholder_text'
    if reason and 'conflicting_' in reason:
        return 'conflicting_component_guess'
    if UNIT_BUILDING.search(remainder):
        return 'unusual_unit_or_building'
    if RURAL_ROAD.search(remainder):
        return 'rural_or_unrecognized_road_form'
    if re.search(r'\d', remainder):
        return 'unrecognized_numeric_segment'
    if not WORD_NUMBER.search(remainder):
        return 'stray_punctuation'
    return 'unrecognized_name_or_location'


def audit(frame, group):
    # Use the same production preprocessing and parser enrichment order as the
    # application, including Romanization, normalization and business-name rules.
    prepared = preprocess_batch(frame)
    result = enrich_address_batch(prepared)
    counts = Counter(rows=result.height)
    raw_addresses = frame['business_address'].fill_null('').to_list()
    counts['raw_addresses_with_six_digit_group'] = sum(bool(re.search(r'(?<!\d)\d{6}(?!\d)', t)) for t in raw_addresses)
    counts['raw_addresses_with_split_pin_shape'] = sum(bool(re.search(r'(?<!\d)[1-9]\d{2}[ -]\d{3}(?!\d)', t)) for t in raw_addresses)
    counts['raw_addresses_with_postal_marker'] = sum(bool(re.search(r'\b(?:pin(?:code)?|postal|postcode|zip)\b', t, re.I)) for t in raw_addresses)
    counts['normalized_addresses_with_six_digit_group'] = 0
    counts['addresses_with_unretained_numeric_groups'] = 0
    counts['parsed_with_street_or_po_box'] = 0
    statuses = Counter()
    coverage = Counter()
    examples = {}
    categories = Counter()
    missing_numbers = Counter()
    total_tokens = 0
    unparsed_tokens = 0
    nonempty_rows = 0
    for row in result.iter_rows(named=True):
        text = row['business_address'] or ''
        if not text.strip():
            counts['empty'] += 1
            continue
        nonempty_rows += 1
        counts['nonempty'] += 1
        counts['normalized_addresses_with_six_digit_group'] += bool(re.search(r'(?<!\d)\d{6}(?!\d)', text))
        status = row['address_parse_status']
        if status == 'parsed' and (row['address_street'] or row['address_po_box']):
            counts['parsed_with_street_or_po_box'] += 1
        statuses[status] += 1
        total_tokens += sum(tokens(text).values())
        for component in COMPONENTS:
            if row['address_' + component]:
                coverage[component] += 1
        remainder = row['address_unparsed_text'] or ''
        if remainder:
            counts['with_unparsed_text'] += 1
            unparsed_tokens += sum(tokens(remainder).values())
            category = categorize(text, remainder, row['address_parse_reason'])
            categories[category] += 1
            example = examples.setdefault(category, [])
            if len(example) < 3:
                example.append({
                    'entity_id': row['entity_id'],
                    'address': text,
                    'recognized': {c: row['address_' + c] for c in COMPONENTS if row['address_' + c]},
                    'unparsed_text': remainder,
                    'parse_reason': row['address_parse_reason'],
                })
        if row['address_parse_status'] == 'partial' and not remainder:
            counts['partial_without_unparsed_text'] += 1
        source_numbers = numbers(text)
        parsed_text = ' '.join((row['address_' + c] or '') for c in COMPONENTS) + ' ' + remainder
        missing = source_numbers - numbers(parsed_text)
        if missing:
            counts['addresses_with_unretained_numeric_groups'] += 1
            missing_numbers.update(missing)
            number_examples = examples.setdefault('unretained_numeric_groups', [])
            if len(number_examples) < 5:
                number_examples.append({
                    'entity_id': row['entity_id'], 'address': text,
                    'missing_numeric_groups': dict(missing),
                    'recognized': {c: row['address_' + c] for c in COMPONENTS if row['address_' + c]},
                    'unparsed_text': remainder,
                })
    counts['status_parsed'] = statuses['parsed']
    counts['status_partial'] = statuses['partial']
    counts['status_fallback'] = statuses['fallback']
    counts['address_word_number_tokens'] = total_tokens
    counts['unparsed_word_number_tokens'] = unparsed_tokens
    counts['unparsed_token_percent'] = round(100 * unparsed_tokens / total_tokens, 2) if total_tokens else 0.0
    return {
        'counts': dict(counts),
        'statuses': dict(statuses),
        'component_nonempty': dict(coverage),
        'unparsed_categories': dict(categories),
        'missing_numeric_groups': dict(missing_numbers),
        'examples': examples,
    }


def main():
    config = load_config()
    report = {
        'run_at_utc': datetime.now(timezone.utc).isoformat(),
        'polars_version': pl.__version__,
        'python_version': sys.version,
        'seed': SEED,
        'per_source': PER_SOURCE,
        'pipeline_order': 'preprocess_batch then country-routed India parse_address',
        'country_aliases': sorted(INDIA),
        'groups': {},
        'files': {},
        'limitations': [
            'Deterministic sampled coverage and text retention are not parser accuracy; no gold component labels are available.',
            'The sample is 10,000 rows per source, selected by entity_id hash seed 42 after filtering country to India.',
            'A complete delivery can now be house, unit, building, street, or PO box; state and city are required. The former parser required street or PO box.',
            'Coverage percentages use nonempty normalized addresses as the denominator.',
            'Word/number token counts ignore punctuation and token order. Canonical city aliases can change lexical spelling.',
            'Unparsed categories are rule-based summaries, mutually exclusive per address, and are not ground truth.',
            'Source hashes and modification times are checked before and after; this audit does not write to source files.',
        ],
    }
    started = time.perf_counter()
    for split, directory in [('train', config.paths.train_data), ('test', config.paths.test_data)]:
        for source_number in (1, 2, 3):
            path = directory / f'{split}_source{source_number}.tsv'
            before = snapshot(path)
            scan = pl.scan_csv(path, separator='\t', infer_schema=False, empty_string_is_null=False)
            selected = (
                scan.filter(pl.col('country').str.to_lowercase().str.strip_chars().is_in(sorted(INDIA)))
                .select('entity_id', 'business_name', 'business_address', 'country')
                .with_columns(pl.col('entity_id').hash(seed=SEED).alias('_sample_hash'))
                .sort('_sample_hash', 'entity_id').head(PER_SOURCE).drop('_sample_hash')
                .collect(engine='streaming')
            )
            key = f'{split}_S{source_number}'
            report['groups'][key] = audit(selected, key)
            report['groups'][key]['sample_entity_ids_sha256'] = hashlib.sha256(
                '\n'.join(selected['entity_id'].to_list()).encode()).hexdigest()
            print(key, report['groups'][key]['counts'], flush=True)
            after = snapshot(path)
            report['files'][str(path)] = {
                'before': before,
                'after': after,
                'unchanged': before == after,
                'sample_rows': selected.height,
            }
    report['elapsed_seconds'] = round(time.perf_counter() - started, 3)
    report['stats'] = {}
    report['component_coverage'] = {}
    report['unparsed_categories'] = {}
    aggregate_counts = Counter()
    aggregate_statuses = Counter()
    aggregate_components = Counter()
    aggregate_categories = Counter()
    aggregate_missing_numbers = Counter()
    for group in report['groups'].values():
        aggregate_counts.update({k: v for k, v in group['counts'].items() if k != 'unparsed_token_percent'})
        aggregate_statuses.update(group['statuses'])
        aggregate_components.update(group['component_nonempty'])
        aggregate_categories.update(group['unparsed_categories'])
        aggregate_missing_numbers.update(group['missing_numeric_groups'])
    aggregate_counts['unparsed_token_percent'] = round(
        100 * aggregate_counts['unparsed_word_number_tokens'] / aggregate_counts['address_word_number_tokens'], 2
    ) if aggregate_counts['address_word_number_tokens'] else 0.0
    report['stats'] = dict(aggregate_counts)
    report['statuses'] = dict(aggregate_statuses)
    report['component_coverage'] = dict(aggregate_components)
    report['unparsed_categories'] = dict(aggregate_categories)
    report['missing_numeric_groups'] = dict(aggregate_missing_numbers)
    report['component_coverage_percent'] = {
        name: round(100 * aggregate_components[name] / aggregate_counts['nonempty'], 2)
        for name in COMPONENTS
    }
    report['parse_status_percent'] = {
        name: round(100 * aggregate_statuses[name] / aggregate_counts['nonempty'], 2)
        for name in ('parsed', 'partial', 'fallback')
    }
    baseline_path = config.paths.artifacts / 'india_pipeline_coverage_60000_before.json'
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        report['baseline_comparison'] = {
            'artifact': baseline_path.name,
            'source_hashes_match': all(
                value['before']['sha256'] == baseline.get('files', {}).get(path, {}).get('before', {}).get('sha256')
                for path, value in report['files'].items()),
            'previous_stats': baseline['stats'],
            'previous_component_coverage': baseline['component_coverage'],
        }
    report['implementation_sha256'] = {
        str(path.relative_to(Path(__file__).resolve().parents[1])): snapshot(path)['sha256']
        for path in [Path(__file__).resolve(), *(
            Path(__file__).resolve().parents[1] / name for name in [
                'src/preprocessing/india.py', 'src/preprocessing/__init__.py',
                'src/preprocessing/address_normalization.py', 'src/resources/india_address.lark',
                'src/resources/india_address_lexicon.json'])]
    }
    report['all_sources_unchanged'] = all(row['unchanged'] for row in report['files'].values())
    destination = config.paths.artifacts / 'india_pipeline_coverage_60000.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('All source files unchanged:', report['all_sources_unchanged'])
    print('Report:', destination, 'seconds:', report['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
