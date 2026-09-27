"""Read-only US comparison; writes only the requested JSON report under artifacts.

Run from project root: .venv/bin/python tests/evaluate_us_grammar.py --per-source 1000
Sampling scans all sources; no address-component gold labels exist for this sample.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import polars as pl
from src.config import load_config
from src.preprocessing.address import enrich_address_batch, US
from src.preprocessing.address_schema import FIELD_NAMES
from src.preprocessing.us import parse_address, lexicon
from src.preprocessing.us_legacy import parse_address as legacy

FIELDS = FIELD_NAMES[:FIELD_NAMES.index('unparsed_text')]


def regression_report():
    fixture=json.loads((ROOT/'tests/fixtures/us_address_regressions.json').read_text())
    result={'description':fixture['description'],'cases':[],'passed':Counter()}
    for text,*values in fixture['cases']:
        expected=dict(zip(('house_number','street','city','state'),values))
        case={'address':text,'expected':expected,'parsers':{}}
        for name,parser in [('us_regex_lark',parse_address),('usaddress',legacy)]:
            actual=asdict(parser(text))
            if actual['house_number']:
                actual['house_number']=re.sub(r'^#\s+', '#', actual['house_number'])
            actual['state']=lexicon()['states'].get(actual['state'],actual['state'])
            mismatches={f:{'expected':v,'actual':actual[f]} for f,v in expected.items() if actual[f]!=v}
            result['passed'][name]+=not bool(mismatches)
            case['parsers'][name]={'mismatches':mismatches,'actual':actual}
        result['cases'].append(case)
    return result


def snapshot(path):
    with path.open('rb') as f:
        digest=hashlib.file_digest(f,'sha256').hexdigest()
    return {'sha256':digest,'size':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns}


def tokens(text):
    return Counter(re.findall(r'[a-z]+|\d+',text.lower()))


def numbers(text):
    return Counter(re.findall(r'\d+(?:[-/]\d+)*',text))


def sample_report(frame):
    rows=frame.to_dicts()
    output={}
    for name,parser in [('us_regex_lark',parse_address),('usaddress',legacy)]:
        # Warm parser initialization before measuring the same sample once.
        parser('123 Main St, Boston, MA')
        start=time.perf_counter()
        parsed=[parser(row['business_address']) for row in rows]
        seconds=time.perf_counter()-start
        counts=Counter(rows=len(rows));statuses=Counter();coverage=Counter();examples=[]
        for row,result in zip(rows,parsed):
            text=row['business_address'] or ''
            statuses[result.parse_status]+=1
            if not text.strip():
                continue
            counts['nonempty']+=1
            parts=' '.join(getattr(result,f) or '' for f in FIELDS)
            combined=parts+' '+(result.unparsed_text or '')
            if result.unparsed_text:
                counts['with_unparsed_text']+=1
            missing=numbers(text)-numbers(combined)
            raw_missing=tokens(text)-tokens(combined)
            counts['missing_numeric_groups_with_remainder']+=bool(missing)
            counts['raw_lexical_flags_with_remainder']+=bool(raw_missing)
            counts['numeric_groups_only_in_remainder']+=bool(numbers(text)-numbers(parts))
            for f in FIELDS:
                coverage[f]+=bool(getattr(result,f))
            if len(examples)<10 and (missing or raw_missing):
                examples.append({'entity_id':row['entity_id'],'address':text,'raw_missing':dict(raw_missing),'missing_numbers':dict(missing),'actual':asdict(result)})
        output[name]={'counts':dict(counts),'statuses':dict(statuses),'coverage':dict(coverage),
                      'parse_seconds':round(seconds,4),'rows_per_second':round(len(rows)/seconds), 'flag_examples':examples}
    enriched=enrich_address_batch(frame)
    output['batch_input_columns_unchanged']=enriched.select(frame.columns).equals(frame)
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--per-source',type=int,default=1000)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/us_grammar_evaluation.json')
    args=parser.parse_args()
    if args.per_source<=0:
        parser.error('--per-source must be positive')
    report={'seed':20260927,'per_source':args.per_source,'regressions':regression_report(),'samples':{},'files':{},
            'limitations':['24 labeled cases are development regressions; not an unbiased accuracy estimate.',
                           'Sampled records have no component gold labels. Coverage/status/retention do not measure correct assignments.',
                           'Raw lexical flags include expected abbreviation, full-state, city-alias and floor-ordinal changes; no lexical equivalence adjustment is applied.',
                           'Numeric retention ignores punctuation except slash/hyphen groups; it is not a semantic correctness check.',
                           'Single local warm-parser timing; includes Python extraction, excludes I/O, normalization and batch assembly.',
                           'US source strings are parsed as supplied; the production pipeline normally runs Roman normalization first.']}
    config=load_config()
    for split,directory in [('train',config.paths.train_data),('test',config.paths.test_data)]:
        for source in (1,2,3):
            path=directory/f'{split}_source{source}.tsv'
            before=snapshot(path)
            scan=pl.scan_csv(path,separator='\t',infer_schema=False,empty_string_is_null=False)
            frame=(scan.filter(pl.col('country').str.to_lowercase().str.strip_chars().is_in(sorted(US)))
                   .select('entity_id','business_address','country')
                   .with_columns(pl.col('entity_id').hash(seed=report['seed']).alias('_sample_hash'))
                   .sort('_sample_hash','entity_id').head(args.per_source).drop('_sample_hash').collect(engine='streaming'))
            name=f'{split}_S{source}'
            report['samples'][name]=sample_report(frame)
            after=snapshot(path)
            report['files'][str(path)]={'before':before,'after':after,'unchanged':before==after}
            print(name,{p:r['counts'] for p,r in report['samples'][name].items() if isinstance(r,dict)},flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('Regression passes:',dict(report['regressions']['passed']),'/ 24',flush=True)
    print('Report:',args.output,flush=True)

if __name__=='__main__':
    main()
