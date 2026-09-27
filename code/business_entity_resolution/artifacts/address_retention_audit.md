# Address retention audit (baseline before city enrichment)

27,000 deterministically sampled records: 18,000 US records across train/test and 9,000 French records from test only. Each eligible source/country contributes the 3,000 lowest seeded entity-ID hashes (seed 42). All six source files were scanned; the audit does not parse every row.

All 45 existing unit tests pass. Separate targeted diagnostics expose 5 failing component expectations out of 7 cases; these are reported findings, not hidden by the unit-suite result. No production parser changes were made.

## Retention results

| Metric | US | France |
| --- | ---: | ---: |
| Sampled rows | 18,000 | 9,000 |
| Nonempty addresses | 17,615 | 8,824 |
| Whole address changed by parser | 0 | 0 |
| Rows missing tokens from components + remainder | 0 | 0 |
| Rows missing numeric groups from components + remainder | 0 | 0 |
| Rows with tokens only retained in remainder | 2,477 | 8,824 |
| Rows with numeric groups only retained in remainder | 593 | 3,763 |

The whole processed address was unchanged for every sampled row. With the unparsed remainder included, no missing alphanumeric tokens or numeric groups were detected. The token test ignores punctuation/order and allows configured street/direction aliases; numeric groups preserve leading zeros and slash/hyphen structure. These checks do not establish correct field assignment.

## Structured extraction gaps

- US: 2,477 of 17,615 nonempty addresses (14.1%) need unparsed text to retain all tested tokens. Reordered city/state segments can remain unassigned. In a targeted case, `floor 3` is assigned to unit rather than floor.
- France: all 8,824 nonempty sampled addresses leave some tokens outside structured fields. House/street are extracted for 4,834; city for only 31; state/region for none. The rules rely on postcode-before-city formatting and miss many actual comma-separated or reordered inputs.
- A targeted French address `12345 rue de la paix, paris` incorrectly yields postcode `12345` and city `rue de la paix`. A real sampled address beginning `17523 r du coq frnacais` shows the same ambiguity.
- A five-digit token anywhere in an address is not proof of a postcode: it can be a house number. Postcode-shaped-token counts in the JSON are diagnostic heuristics, not postcode recall.

## Targeted checks

| Country | Address | Result |
| --- | --- | --- |
| US | 123 main st suite 004, boston ma 02108 | Pass |
| US | oh, waverly, 613 market st | city: expected 'waverly', got None; state: expected 'oh', got None |
| US | 123 main st floor 3, boston ma 02108 | floor: expected 'floor 3', got None |
| France | 12 bis rue de la paix, 75002 paris | Pass |
| France | 105 quai des chevillards, lille, hauts-de-france | city: expected 'lille', got None; state: expected 'hauts-de-france', got None |
| France | 12345 rue de la paix, paris | house_number: expected '12345', got None; street: expected 'rue de la paix', got None; city: expected 'paris', got 'rue de la paix'; postal_code: expected None, got '12345' |
| France | 0026 r. des pyramides, roubaix, nord | house_number: expected '0026', got None; city: expected 'roubaix', got None |

## Implications

Keep whole-address and unparsed-text matching features; component-only matching would discard usable evidence. Improve French structure handling and numeric disambiguation before relying on its fields for blocking. Correct US floor routing and evaluate reordered segments. Libpostal remains unavailable and was not evaluated. These are follow-up items, not changes made by this audit.

All source-file SHA-256 hashes and modification times were unchanged. The audit begins after Roman text normalization; it does not measure information lost during earlier transliteration or normalization. Without gold component annotations, correct extraction accuracy and downstream matching impact remain unmeasured.

Reproduce from the project directory:
```bash
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python tests/audit_address_retention.py --per-group 3000
```

Detailed counts, capped examples, expectations and checksums: [address_retention_audit.json](address_retention_audit.json).

The JSON audit output has since been refreshed after city enrichment. For current coverage and alias-normalization caveats, see the main README, “User-supplied city dictionaries”. This Markdown report retains the baseline results.
