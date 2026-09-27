"""Reproducible read-only sampled retention audit (not a parser accuracy score).

Run: .venv/bin/python tests/audit_address_retention.py --per-group 3000
Scans all six sources; selects lowest seeded ID hashes per source/country.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import polars as pl
from src.config import load_config
from src.preprocessing.address import enrich_address_batch, parse_address, US, FRANCE
from src.preprocessing.address_schema import FIELD_NAMES
from src.preprocessing.text import romanize_series, normalized_expr
from src.preprocessing.us import _aliases
from src.preprocessing import us, france
from src.preprocessing.cities import city_index, key as city_key
from src.preprocessing.address_normalization import rules

COMPONENTS = FIELD_NAMES[:FIELD_NAMES.index('unparsed_text')]
ALIASES = {**_aliases()['directions'], **_aliases()['street_types']}
# Canonicalize French aliases through the same token equivalence map.
ALIASES.update({key: ALIASES.get(value, value) for key, value in rules()['france']['street_types'].items()})

def words(text):
    return Counter(ALIASES.get(t,t) for t in re.findall(r'[a-z]+|\d+', text.lower()))

def numbers(text):
    return Counter(re.findall(r'\d+(?:[-/]\d+)*', text))

def alias_adjusted_missing(text, combined, row):
    """Allow only a documented replacement of the base parser's city field.

    Preserve the raw metric too. This assesses retention, not whether a city
    mapping is geographically correct. Do not equate aliases across whole text.
    """
    missing = words(text) - words(combined)
    if not missing:
        return missing
    country = (row['country'] or '').strip().lower()
    base = us._parse_address(text) if country in US else france._parse_address(text)
    new_city = row['address_city']
    if not base.city or not new_city or base.city == new_city:
        return missing
    index, _ = city_index('us' if country in US else 'france')
    if new_city not in {city for city, context in index.get(city_key(base.city), [])}:
        return missing
    # Restore exactly this city's pre-canonicalization value for lexical audit.
    equivalent = ' '.join((base.city if c == 'city' else row['address_'+c]) or '' for c in COMPONENTS)
    equivalent += ' ' + (row['address_unparsed_text'] or '')
    return words(text) - words(equivalent)


def snapshot(path):
    with path.open('rb') as f:
        digest = hashlib.file_digest(f,'sha256').hexdigest()
    return {'sha256':digest,'mtime_ns':path.stat().st_mtime_ns}


def audit(frame):
    normalized=frame.with_columns(romanize_series(frame['business_address']).alias('business_address'))
    normalized=normalized.with_columns(normalized_expr('business_address',address=True))
    result=enrich_address_batch(normalized)
    stats=Counter(rows=result.height)
    status=Counter(); coverage=Counter(); examples={}
    for row in result.iter_rows(named=True):
        text=row['business_address'] or ''
        status[row['address_parse_status']]+=1
        if not text.strip():
            stats['empty']+=1
            continue
        stats['nonempty']+=1
        parts=' '.join(row['address_'+c] or '' for c in COMPONENTS)
        remainder=row['address_unparsed_text'] or ''
        combined=parts+' '+remainder
        for c in COMPONENTS:
            if row['address_'+c]: coverage[c]+=1
        checks={
            'missing_tokens_components_only': words(text)-words(parts),
            'missing_tokens_with_remainder': words(text)-words(combined),
            'missing_tokens_with_remainder_alias_adjusted': alias_adjusted_missing(text, combined, row),
            'missing_numbers_components_only': numbers(text)-numbers(parts),
            'missing_numbers_with_remainder': numbers(text)-numbers(combined),
        }
        for key,missing in checks.items():
            if missing:
                stats[key]+=1
                if len(examples.setdefault(key,[]))<8:
                    examples[key].append({'entity_id':row['entity_id'],'address':text,
                        'missing':dict(missing),'fields':{c:row['address_'+c] for c in COMPONENTS if row['address_'+c]},
                        'unparsed_text':remainder,'status':row['address_parse_status']})
        if re.search(r'\d',text) and not row['address_house_number'] and not row['address_po_box']:
            stats['numeric_address_without_house_or_po_box']+=1
        zip_matches=re.findall(r'(?<![\w-])\d{5}(?:-\d{4})?(?![\w-])',text)
        if len(zip_matches)==1:
            stats['single_postcode_shaped_token']+=1
            if row['address_postal_code']!=zip_matches[0]: stats['postcode_shaped_token_not_in_postal_field']+=1
        if re.search(r'\b(?:apt|apartment|suite|ste|unit|floor|fl|etage|appartement|batiment)\b|#',text):
            stats['unit_floor_building_marker']+=1
            if not any(row['address_'+c] for c in ('unit','floor','building')):
                stats['marker_without_unit_floor_building_field']+=1
    stats['whole_address_changed_by_parser']=int((result['business_address']!=normalized['business_address']).fill_null(False).sum())
    return {'counts':dict(stats),'statuses':dict(status),'component_nonempty':dict(coverage),'examples':examples}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--per-group',type=int,default=3000)
    args=parser.parse_args()
    if args.per_group<=0:parser.error('--per-group must be positive')
    config=load_config();report={'seed':42,'per_group':args.per_group,'groups':{},'files':{}}
    start=time.perf_counter()
    for split,directory in [('train',config.paths.train_data),('test',config.paths.test_data)]:
        for n in (1,2,3):
            path=directory/f'{split}_source{n}.tsv';before=snapshot(path)
            scan=pl.scan_csv(path,separator='\t',infer_schema=False,empty_string_is_null=False)
            countries=[('US',US)] + ([('France',FRANCE)] if split=='test' else [])
            for label,aliases in countries:
                selected=(scan.filter(pl.col('country').str.to_lowercase().str.strip_chars().is_in(sorted(aliases)))
                    .select('entity_id','business_address','country')
                    .with_columns(pl.col('entity_id').hash(seed=42).alias('_sample_hash'))
                    .sort('_sample_hash','entity_id').head(args.per_group).drop('_sample_hash').collect(engine='streaming'))
                key=f'{split}_S{n}_{label}'
                report['groups'][key]=audit(selected)
                print(key,report['groups'][key]['counts'],flush=True)
            after=snapshot(path)
            report['files'][str(path)]={'before':before,'after':after,'unchanged':before==after}
    cases = [
        ('US', '123 main st suite 004, boston ma 02108', {'house_number':'123', 'unit':'suite 004', 'city':'boston', 'state':'ma', 'postal_code':'02108'}),
        ('US', 'oh, waverly, 613 market st', {'house_number':'613','city':'waverly','state':'oh'}),
        ('US', '123 main st floor 3, boston ma 02108', {'floor':'floor 3'}),
        ('France', '12 bis rue de la paix, 75002 paris', {'house_number':'12 bis','city':'paris','postal_code':'75002'}),
        ('France', '105 quai des chevillards, lille, hauts-de-france', {'house_number':'105','city':'lille','state':'hauts-de-france'}),
        ('France', '12345 rue de la paix, paris', {'house_number':'12345','street':'rue de la paix','city':'paris','postal_code':None}),
        ('France', '0026 r. des pyramides, roubaix, nord', {'house_number':'0026','city':'roubaix'}),
    ]
    report['targeted_cases'] = []
    for country,address,expected in cases:
        actual=asdict(parse_address(address,country))
        mismatches={key:{'expected':value,'actual':actual[key]} for key,value in expected.items() if actual[key]!=value}
        report['targeted_cases'].append({'country':country,'address':address,'expected':expected,
            'actual':actual,'mismatches':mismatches,'passed':not mismatches})
    report['elapsed_seconds']=round(time.perf_counter()-start,3)
    report['limitations']=[
        'Sampled retention, not full-dataset accuracy; no gold address-component labels.',
        'Whole-address comparison starts AFTER Roman normalization, so does not assess earlier transliteration loss.',
        'Token comparisons ignore punctuation/order and canonicalize US street/direction aliases globally.',
        'Alias-adjusted lexical checks restore only known canonicalized parser-city fields; raw metrics are also retained.',
        'Numeric-span comparison retains slash/hyphen groups and leading zeros; differences require inspection.',
        'Postcode-shaped and occupancy-marker checks are heuristics, not ground truth.',
        'Project-owned French rules only; external address backends are not selected.'
    ]
    destination=config.paths.artifacts/'address_retention_audit.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print('Report:',destination,'seconds:',report['elapsed_seconds'],flush=True)

if __name__=='__main__':main()
