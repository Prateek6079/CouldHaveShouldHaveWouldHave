## Indian parser coverage audit

Ran `preprocess_batch` normalization followed by country-routed India parsing on **60,000 Indian records**, selected deterministically using entity-ID hash seed 42, 10,000 from each train/test source. Run started 2026-09-27T12:38:10.359379+00:00; Polars 1.44.2.

| Result (among nonempty addresses) | Count | Percentage |
| --- | ---: | ---: |
| Nonempty addresses | 58,861 | 100.00% |
| Fully structured parses | 11,266 | 19.14% |
| Partial parses | 47,227 | 80.23% |
| Fallbacks (no component extracted) | 368 | 0.63% |
| Addresses containing leftover text | 45,543 | 77.37% |
| Partial parses without leftover text | 2,052 | 3.49% |

There were **1,139 empty addresses**. The nonempty normalized addresses contained **695,708 word/number tokens**, of which **212,377 (30.53%)** remained in `unparsed_text`, down from **52.60%** in the saved baseline. Token counts ignore punctuation and order. Contextual PIN normalization joins split numeric groups, so the normalized token denominator changes slightly.

| Component | Count | Coverage | Previous coverage |
| --- | ---: | ---: | ---: |
| State | 52,567 | 89.31% | 90.99% |
| Street | 18,311 | 31.11% | 28.43% |
| House number | 37,517 | 63.74% | 21.79% |
| City | 49,849 | 84.69% | 51.63% |
| Locality | 36,475 | 61.97% | 28.68% |
| Unit | 8,382 | 14.24% | 10.74% |
| Floor | 7,284 | 12.37% | 12.33% |
| Building | 10,769 | 18.30% | 10.83% |
| Landmark direction | 6,329 | 10.75% | 10.63% |
| PO box | 20 | 0.03% | 0.03% |
| Postal code | 90 | 0.15% | 0.00% |

**Numeric retention:** 0 addresses lost numeric groups between normalized input and extracted fields plus leftovers (previously 17). This measures normalized-input retention, not byte-for-byte raw-text equality.

**PIN diagnosis:** the raw sample had 12 addresses containing six-digit sequences and 790 containing a three-plus-three digit shape. Most split shapes are premise ranges, not PINs. Contextual normalization exposes 99 addresses with six-digit groups; the parser extracts PINs from 90 addresses. Examples fixed include `Chennai600004`, `Pune411014`, and `Madurai-625 001`. Unit IDs and registration numbers are not automatically classified as PINs.

**Status interpretation and comparison:** partial parsing remains enabled. `parsed` requires city, state, a delivery component, accepted granular-to-large order, no conflicts, and no leftover text. A delivery component can now be a house, unit, building, street, or PO box; the baseline required street or PO box. Consequently, the headline fully structured count is not directly comparable to the old 506. Under the old delivery requirement, **3,207 (5.45%)** current parses also contain a street or PO box. `partial` retains extracted fields when incomplete, conflicting, reordered, or accompanied by leftovers; `fallback` extracts no component; empty input is `empty`.

| Largest leftover category (heuristic) | Addresses |
| --- | ---: |
| unrecognized name or location | 18,914 |
| unrecognized numeric segment | 14,914 |
| conflicting component guess | 4,912 |
| unusual unit or building | 4,848 |
| placeholder text | 1,257 |
| rural or unrecognized road form | 638 |
| stray punctuation | 60 |

One category is assigned per address containing leftovers; these labels are diagnostics rather than ground truth.

| Regression address | Parser-assigned fields | Leftover text |
| --- | --- | --- |
| `303, Krishna Puri Main Road, Mandawali, Delhi` | house number `303`; street `krishna puri main road`; locality `mandawali`; city `delhi`; state `delhi` | None |
| `Plot No. 20B, Gaur Bheet Faizullaganj, Lucknow, Uttar Pradesh` | house number `plot 20b`; locality `gaur bheet faizullaganj`; city `lucknow`; state `uttar pradesh` | None |
| `17/18, III Floor S K C Buildings, Mill Road, Coimbatore, Tamil Nadu` | street `mill road`; unit `17/18`; floor `floor 3`; building `s k c buildings`; city `coimbatore`; state `tamil nadu` | None |

**Validation:** all 27 targeted India regression tests pass. The full suite runs 95 tests: 94 pass, with one pre-existing US failure (`test_us_grammar.USGrammarTests.test_floors`, `10th Fl.`). The obsolete India-deferred assertion was updated to test the active India route.

**Limitations:** these are coverage figures, not accuracy scores; the dataset has no gold component labels. Unmarked locality/city guesses and bare-number unit-versus-house decisions remain heuristic. A leading bare ID defaults to house, or unit if a floor follows or an explicit house number exists. Ambiguous or repeated components are retained as leftovers. State/city disambiguation has no geographic consistency validation, and unknown or corrupted names remain challenging.

All six source files retained their SHA-256 hashes and modification times. Baseline source hashes match this run: **True**. The report stores per-source selected-ID digests, environment versions, and implementation hashes; Polars hashing is version-dependent.

Reproduce from `code/business_entity_resolution`:

```bash
.venv/bin/python -m unittest discover -s tests -p test_india.py -v
.venv/bin/python tests/audit_india_pipeline_60000.py
```

Detailed results: [current JSON](india_pipeline_coverage_60000.json), [saved baseline](india_pipeline_coverage_60000_before.json), [audit script](../tests/audit_india_pipeline_60000.py), [India regression tests](../tests/test_india.py).
