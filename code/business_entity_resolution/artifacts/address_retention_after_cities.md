# Address retention rerun after city enrichment

All 56 unit tests pass. Audit: same seeded 27,000-record sample (18,000 US train/test; 9,000 French test only), 3,000 records per source/country. No production parser changes in this rerun.

| Check | US | France |
| --- | ---: | ---: |
| Nonempty addresses | 17,615 | 8,824 |
| Whole processed address changed | 0 | 0 |
| Raw missing-token flags (before city equivalence) | 32 | 0 |
| Missing-token flags allowing documented city substitutions | 0 | 0 |
| Missing numeric groups including remainder | 0 | 0 |
| Rows needing remainder for tokens (raw comparison) | 2,482 | 8,322 |
| Rows needing remainder for numeric groups | 593 | 2,523 |

The 32 raw US lexical flags are explained by documented substitutions of an existing city field. The adjusted audit restores only that known pre-canonicalization field for comparison; it does not replace arbitrary aliases across the address. A negative-control check confirmed it still detects a deliberately removed house number.

| Field present | US | France |
| --- | ---: | ---: |
| house_number | 16,471 | 6,092 |
| street | 16,940 | 6,092 |
| city | 16,090 | 3,873 |
| state | 15,217 | 0 |
| postal_code | 11 | 48 |
| unit | 1,504 | 0 |
| floor | 1 | 0 |
| building | 254 | 0 |

Compared with the baseline, city-field coverage increased from 15,868 to 16,090 for US and 31 to 3,873 for France. The French baseline comparison also includes the intervening street-abbreviation change. These are presence counts, not accuracy.

Whole addresses and all six input-file hashes/modification times stayed unchanged. Token checks ignore punctuation/order; numeric checks retain leading zeros and slash/hyphen groups. Earlier transliteration/normalization is outside this audit. Some information remains solely in unparsed text, and known postcode, city/state and floor assignment errors remain. Retention does not demonstrate correct component assignment or downstream model quality.

## Targeted diagnostics

- US: `123 main st suite 004, boston ma 02108` — pass
- US: `oh, waverly, 613 market st` — mismatches: {"city": {"expected": "waverly", "actual": null}, "state": {"expected": "oh", "actual": null}}
- US: `123 main st floor 3, boston ma 02108` — mismatches: {"floor": {"expected": "floor 3", "actual": null}}
- France: `12 bis rue de la paix, 75002 paris` — pass
- France: `105 quai des chevillards, lille, hauts-de-france` — mismatches: {"state": {"expected": "hauts-de-france", "actual": null}}
- France: `12345 rue de la paix, paris` — mismatches: {"house_number": {"expected": "12345", "actual": null}, "street": {"expected": "rue de la paix", "actual": null}, "city": {"expected": "paris", "actual": "rue de la paix"}, "postal_code": {"expected": null, "actual": "12345"}}
- France: `0026 r. des pyramides, roubaix, nord` — mismatches: {"city": {"expected": "roubaix", "actual": null}}

Reproduce: `.venv/bin/python tests/audit_address_retention.py --per-group 3000` from the project directory. [Detailed JSON](address_retention_audit.json).
