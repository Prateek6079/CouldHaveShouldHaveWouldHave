# Source layout

## Technology reference

Before implementing a component, consult the
[agreed technology stack and implementation rules](../README.md#agreed-technology-stack).
The core is **Polars + Lark + scikit-learn/SciPy + RapidFuzz + LightGBM**, with
Parquet intermediates, NumPy feature arrays, sparse-dot-topn retrieval and
regex + Lark for US/Indian parsing (usaddress is retained only for comparisons). Project-owned rules handle France; libpostal was evaluated on a US sample and not adopted. Learned transliteration remains a conditional evaluation. The parent README is the authoritative stack
reference; add dependencies only when their component is implemented.

Configuration, read-only TSV validation, and Roman text preprocessing are implemented.
US/French/Indian/generic address adapters are implemented; later ML stages remain
scaffolds. Legal-form, US address, and Indian address lexicons are populated; city aliases
cover supplied US/French and common Indian cities. End-to-end execution is unavailable.

| Module/directory | Responsibility |
| --- | --- |
| `__main__.py` | Command-line entry point |
| `pipeline.py` | Connect processing stages |
| `config.py`, `configs/` | Configuration loading and settings |
| `data_io.py` | TSV input and schema validation |
| `preprocessing/` | Normalization, transliteration, names, and address adapters |
| `resources/` | Legal suffixes, address aliases, and city aliases |
| `blocking.py` | Candidate retrieval and pair deduplication |
| `features.py` | Shared training/inference feature computation |
| `labels.py`, `splitting.py` | Ground-truth labeling and entity-level splits |
| `training.py`, `evaluation.py` | Model fitting, validation, and threshold selection |
| `inference.py` | Candidate scoring and match selection |
| `submission.py` | Final matching-pair and submission TSV export |

Configuration paths are defined relative to `code/business_entity_resolution/`,
independently of the shell working directory. The configuration loader implements
this convention for both config filenames and configured paths.

Runtime artifacts belong in `code/business_entity_resolution/artifacts/`:
processed data, candidate pairs, features, fitted models, validation metrics,
and logs. Create this directory when the pipeline is implemented and run.

The repository-level `output/` is reserved exclusively for final results and
matching pairs. Configuration, caches, and trained models must not go there.

Use the parent `README.md` for reproducible end-to-end execution instructions
once implemented. Use repository `Documentation_template.md` for methodology.

## Component 1: configuration (complete)

`config.load_config()` returns immutable `PipelineConfig` and `PathsConfig`
objects with absolute `Path` values. It accepts an optional JSON config path,
validates required/unknown keys and a 32-bit unsigned integer seed, and rejects
overlap between output directories or between outputs and input directories.
Invalid settings raise `ConfigError`. Loading creates no directories and does
not check input existence; those checks belong to input validation.

Run tests from `code/business_entity_resolution/`:

```bash
python3 -m unittest discover -s tests -v
```

## Component 2: read-only TSV validation and Polars batches (complete)

- `read_source(path, source, batch_size=100_000)` yields Polars DataFrames with
  string columns, preserving original text, extra columns and row order.
- `read_ground_truth(path, batch_size=100_000)` yields string reference IDs and
  `List(String)` matching IDs. Empty match cells become empty lists.
- `load_dataset(config, "train" | "test", batch_size=100_000)` validates a split
  and returns re-iterable `BatchTable` readers and `row_counts`. Test has no
  ground truth. Consuming the readers rereads inputs after the validation pass.

Inputs are opened only for reading; no files or directories are created. The
strict CSV parser handles TSV quoting and UTF-8/BOM; Polars provides compact
batch tables. Invalid row widths, missing/duplicate/blank column names, blank
records, invalid/duplicate IDs and invalid match lists raise `DataValidationError`.
Duplicate checks span batches. Extra columns, empty text and unfamiliar countries
are valid. Training requires complete S1 ground truth and existing S2/S3 matches.

Consume batches without accumulating them. Full-file validation requires full
iteration; close a reader when stopping early. Header-only files yield no batches.
Inputs must remain stable between validation and processing. Batch text is bounded,
but exact validation retains ID indexes proportional to file size.

Install pinned Polars from the parent `requirements.txt` in the project virtual
environment, then run `python -m unittest discover -s tests -v` from the project
directory. See the parent README for usage and measured memory/runtime results.
The pipeline CLI remains a scaffold. US/French/Indian/generic address adapters are described below; the pipeline CLI remains a scaffold.

## Component 3: Roman text preprocessing (complete)

`preprocess_batch(frame)` returns a new Polars frame; `preprocess_batches(frames)`
yields one processed batch at a time. Names and addresses are replaced with one
Roman version. No original-script/full-name/legal-form columns are appended.
Original TSVs remain untouched; IDs, countries, extra columns and order persist.

- `transliteration.py`: NFKC cleanup, supported Indic script-run conversion via
  indic-transliteration, modern-character compatibility mappings, then Latin
  accent folding. Unsupported letters raise `UnsupportedScriptError`.
- `text.py`: batch-local conversion of distinct non-ASCII values; native Polars
  lowercase/whitespace/punctuation processing. Address unit/range separators stay.
- `business_name.py`: boundary legal-form removal using version 1 of
  `resources/legal_suffixes.json`, including country-specific forms.
- `__init__.py`: validate the batch schema and compose these operations.

Install `requirements.txt` into the project `.venv` (Polars 1.44.2 and
indic-transliteration 2.3.82). See the parent README's
[preprocessing contract](../README.md#batch-text-preprocessing) for examples,
supported scripts, dictionary schema, limitations and evaluation requirements.
This step performs no file writes, parsing, blocking, training or CLI orchestration.

## Component 4: address adapters (US/French/Indian/generic)

- `address_schema.py`: immutable nullable `AddressResult` and shared column names.
- `address.py`: country routing and Polars batch enrichment with a bounded local
  cache. Appends `address_*` columns and preserves whole processed addresses.
- `us.py`: default regex + Lark parser. Extracts independent unit/floor/building,
  house/street, locality/city/state, ZIP/PO-box and landmark spans, then checks
  their assembly. Missing pieces never discard recognized components.
- `component_grammar.py`: token offsets, non-overlapping spans, retained unknown
  text and process-cached Lark LALR grammars for US and India.
- `resources/us_address_lexicon.json`: version 1 US vocabulary and aliases, full
  state/code dictionary, bounded-name setting and small building/locality lists.
- `resources/india_address_lexicon.json`: version 1 Indian vocabulary for address
  numbers, locality/admin markers, road forms, canonical state/UT aliases
  and six-digit PINs; it is loaded by the India adapter.
- `resources/us_address.lark`: component rules plus common normal/reverse orders.
- `us_legacy.py`: preserved usaddress adapter for explicit comparisons only.
- `france.py`: existing project-owned partial extraction; unchanged by US work.
- `india.py`: Indian lexical extraction, rural/urban assembly and partial fallback.
- `generic.py`: retain unknown text for unsupported countries.

Use `enrich_address_batches(preprocess_batches(read_source(...)))`. No files are
written. The default US `parser` value is `us_regex_lark`. Missing fields are null;
`parsed` requires complete core components, recognized order and no leftover text.
`partial` preserves independent matches; `fallback` retains the whole unknown
address. A parse status is not a confidence score. Conflicting duplicate fields
stay in `unparsed_text` with a reason. `landmark_direction` holds relation and
name together; street directions remain inside `street`.

Lark 1.3.1 is installed in the project `.venv` and pinned. Its grammar and the
lexicon load once per process; restart after editing resources. The existing
usaddress dependency is used only by the comparison adapter and its tests.

### Best-guess normalization and city assistance

The US lexicon canonicalizes recognized street types, street directions, unit
markers, floor ordinals and full state names. `cities.py` supplies the existing
50-city US index; explicit state context constrains city aliases. Known cities
are recognized in remaining spans; a short unknown city requires location
context. The city dictionary never invents a missing state. This is separate
from French enrichment, which continues to use its 50-commune list and Réunion
context checks. `address_normalization.py` and `address_aliases.json` remain in
use by France and the legacy US adapter.

The shared `postal_code` output covers contextual five-digit ZIP and ZIP+4
values. Explicit `zip`, `zip code`, `zipcode`, `postal code`, `pin code` and
`pincode` labels are also recognized and removed from leftover text; leading
zeros in the code are preserved.

For example, `500 Park Avenue Suite 400 New York NY` becomes house `500`, street
`park ave`, unit `suite 400`, city `new york city`, state `ny`. For
`Suite 420, Jefferson Tower, Somewhere near Central Park`, recognized unit,
building and landmark survive while `Somewhere` stays unparsed.

### Validation and known limits

See the parent [US grammar implementation and evaluation](../README.md#custom-us-regex--grammar-parser)
for exact rules, comparison results and commands. Specification and regression
tests are in `tests/test_us_grammar.py`; a reproducible read-only comparison is
`tests/evaluate_us_grammar.py`. The parser-only comparison is
[`artifacts/us_grammar_coverage_60000.json`](../artifacts/us_grammar_coverage_60000.json),
and the actual preprocessing-to-parser run is
[`artifacts/us_pipeline_coverage_60000.json`](../artifacts/us_pipeline_coverage_60000.json).
Both produced the same component/status counts: 72.83% of nonempty rows were
structurally parsed, 27.17% were partial, and
17.93% retained some `unparsed_text` (7.64% of sampled word/number tokens).
Component plus remainder retained every tested numeric group. All source hashes
and modification times were unchanged. The earlier
[city retention audit](../artifacts/address_retention_after_cities.md) is
historical and describes the previous adapters.

Unlisted street types, numeric-only/rural roads and locality-versus-city ambiguity
remain gaps. One field value is retained per component; conflicts and unfamiliar
segments remain text. The grammar bounds name spans to 16 tokens; unknown city
candidates to five. Keep whole-address similarity and missingness/parse-status
features in later matching. Coverage and retention do not measure correct labels
or downstream matching quality. France and India use country-specific rules and grammars; Indian component
city-alias coverage and extraction accuracy remain unevaluated.
