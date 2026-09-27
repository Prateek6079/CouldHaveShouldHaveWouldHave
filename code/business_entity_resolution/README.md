# Business Entity Resolution

Match business records from Source 2 and Source 3 to the deduplicated reference records in Source 1. A Source 1 entity can have zero, one, or many matching records across the other two sources.

**Status: configuration, read-only TSV validation, batch text preprocessing and US/French/Indian/generic address adapters are implemented.** Dependencies are pinned in `requirements.txt`; blocking, model training, inference and submission generation remain unimplemented. The end-to-end workflow is still a plan.

## Agreed technology stack

This is the implementation reference for subsequent components. Build a CPU-first
pipeline in this order: explicit parsing → candidate blocking → shared feature
engineering → classification. Keep these choices when implementing new stages;
record any agreed changes here. Selection does not mean a dependency is already
installed or a component implemented. Add and pin dependencies as their stages
are implemented. Current direct dependencies are Polars, indic-transliteration, Lark and usaddress (comparison adapter only).

| Stage | Selected technology | Purpose |
| --- | --- | --- |
| Data loading, transformations and joins | **Polars** | Compact batch tables; native expressions, lazy scans and streaming where suitable |
| Intermediate storage | **Parquet** | Enriched records, candidates and features under `artifacts/` |
| Basic normalization | **`unicodedata`, `re`, native Polars expressions** | Unicode, whitespace, punctuation, markers and aliases |
| Indian address parsing | **Lark** | Explicit grammar with a separate vocabulary; start with LALR, evaluate Earley only if ambiguity requires it |
| US address parsing | **Regex + Lark LALR** | Shallow component grammar and US lexicon; usaddress retained only for explicit comparisons |
| French address parsing | **Project-owned conservative rules** | Partial deterministic extraction; improve using observed parsing errors |
| Indic transliteration | **`indic-transliteration`** | Convert supported Indic script runs; evaluate IndicXlit later only if matching quality requires it |
| Structured candidate retrieval | **Polars joins** | Postal-code, city and other structured retrieval routes |
| Name-based candidate retrieval | **scikit-learn + SciPy + sparse-dot-topn** | Character TF-IDF, CSR sparse matrices and top-k similarity retrieval |
| String comparison features | **RapidFuzz** | Optimized edit/token similarity and corresponding-pair `cpdist` comparisons |
| Numerical feature arrays | **NumPy** | Compact numeric matrices for candidate pairs |
| Matching classifier | **LightGBM (`LGBMClassifier`)** | First binary classifier; XGBoost is an optional later comparison, not an undecided default |
| Validation | **scikit-learn + custom evaluation** | Pair metrics, candidate recall, threshold selection and complete submission-level evaluation |
| CLI, configuration and logging | **`argparse`, `json`, `logging`** | Lightweight standard-library orchestration |

### Implementation rules

- Keep original inputs read-only. Preserve IDs and postal codes as strings.
  Use batches throughout, avoid full Python row-dictionary collections and
  repeated row-wise dataframe operations, and save intermediates as Parquet
  under `artifacts/`. Final deliverables remain TSV under repository `output/`.
- Prefer native Polars transformations, joins and lazy/streaming execution where
  supported. Current ingestion deliberately retains Python's strict TSV parser
  before creating Polars batches to preserve malformed-row validation. Custom
  Python grammar parsing does not become faster merely by using Polars tables.
- Parse each record once before blocking. For India, keep grammar in
  `src/resources/india_address.lark`, vocabulary in
  `src/resources/india_address_lexicon.json`, and parser initialization,
  token preparation, component conversion and fallback in `preprocessing/india.py`.
  The India parser loads its versioned lexicon and LALR grammar once per process.
  Extend both from observed retained-text errors; test city inference carefully
  because a comprehensive Indian gazetteer is not included.
- Country adapters must share a component schema, preserve unparsed text and
  report parse status. Support partial extraction and generic fallback; parser
  failure must not prevent matching. French accent handling uses Unicode
  normalization. Learned transliteration is optional, not a default per-record step.
- Combine structured blocks with name retrieval. Retrieve S2 and S3 separately,
  then union and deduplicate pairs. Store pair IDs or integer row indices and
  join needed fields in batches. Monitor large blocks; top-k limits retained
  pairs but does not bound all sparse multiplication work.
- Compute edit, token-sort/token-set, Jaccard, TF-IDF cosine, component agreement
  and conflict, numeric, missingness and parse-status features. Use corresponding-
  pair comparisons, not full Cartesian distance matrices. Keep token-set scores
  alongside other evidence. Avoid a fixed country encoding; prioritize signals
  that transfer to unseen countries.
- Share feature computation between training and inference. Split by reference
  entity, fit learned state only on training partitions, and tune thresholds on
  complete validation candidates rather than sampled negatives. Count matches
  missed by blocking when evaluating final quality.
- Use native library parallelism. Benchmark batched multiprocessing for Python
  parsing and coordinate thread counts to avoid oversubscription. Persist the
  model, feature order, vectorizers, threshold, configuration and lexical/grammar
  versions together for reproducible prediction.

Before committing to settings, measure parser throughput/component coverage,
candidate recall versus volume/runtime, feature throughput/peak memory, and
final matching quality under the official metric. The selected stack is a design
choice, not a claim that every library has been benchmarked on these inputs.
IndicXlit, Earley and XGBoost remain conditional evaluations. The team evaluated libpostal on a US sample and chose project-owned country parsers; libpostal is not a dependency or runtime backend.

## Repository layout

```text
simple_ml_pipeline/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── code/
│   └── business_entity_resolution/
│       ├── README.md
│       ├── requirements.txt
│       ├── artifacts/                 # Future runtime directory
│       └── src/
│           ├── README.md              # Short source-layout guide
│           ├── __init__.py
│           ├── __main__.py
│           ├── config.py
│           ├── configs/
│           │   └── default.json
│           ├── data_io.py
│           ├── preprocessing/
│           │   ├── __init__.py
│           │   ├── text.py
│           │   ├── transliteration.py
│           │   ├── business_name.py
│           │   ├── address.py
│           │   ├── us.py
│           │   ├── india.py
│           │   ├── france.py
│           │   └── generic.py
│           ├── resources/
│           │   ├── legal_suffixes.json
│           │   ├── address_aliases.json
│           │   ├── city_aliases.json
│           │   ├── us_address.lark
│           │   ├── us_address_lexicon.json
│           │   ├── india_address.lark
│           │   └── india_address_lexicon.json
│           ├── blocking.py
│           ├── features.py
│           ├── labels.py
│           ├── splitting.py
│           ├── training.py
│           ├── evaluation.py
│           ├── inference.py
│           ├── submission.py
│           └── pipeline.py
├── output/                            # Final results and matching pairs only
└── Documentation_template.md           # Methodology and experimental findings
```

All pipeline code, configuration, and lexical resources live in `src/`. Original data stays in `dataset/`. The project-level `artifacts/` directory is intended for future pipeline runs and has not been created by the scaffold.

This README documents the layout and execution workflow. [Documentation_template.md](../../Documentation_template.md) is for the methodology, design decisions, experiments, results, and limitations. [src/README.md](src/README.md) provides a shorter module index.

## Input data

Read all challenge files as tab-separated values, not comma-separated CSV. Addresses and matching-ID lists can contain commas.

| Source-file column | Meaning |
| --- | --- |
| `entity_id` | Unique record ID; the `S1-`, `S2-`, or `S3-` prefix indicates its source |
| `business_name` | Business name with possible spelling and formatting variations |
| `business_address` | Address with possible missing or inconsistent components |
| `country` | Unrestricted country string |

| Ground-truth column | Meaning |
| --- | --- |
| `source1_entity_id` | Reference entity ID |
| `matched_entity_ids` | Comma-separated matching Source 2/3 IDs; empty for no matches |

Training contains US and India records. Test additionally contains France. Keep country handling open to unfamiliar labels; every test Source 1 entity must appear in the submission.

## Module responsibilities

### Entry point and shared infrastructure

| File | Intended responsibility |
| --- | --- |
| `__init__.py` | Mark `src` as a Python package |
| `__main__.py` | Handle CLI arguments and delegate execution; currently exits with a scaffold-only message |
| `pipeline.py` | Connect stages, manage artifacts, and control execution order |
| `config.py` | Load and validate configuration and resolve paths |
| `configs/default.json` | Store default paths and random seed |
| `data_io.py` | Read TSV files and validate required columns, unique IDs, and source prefixes |

Keep CLI handling separate from processing logic. Stage functions should accept explicit inputs and return results or artifact references so they can be reused independently.

### Preprocessing

| File | Intended responsibility |
| --- | --- |
| `preprocessing/__init__.py` | Batch preprocessing API; replace name/address text without duplicate representations |
| `preprocessing/text.py` | Normalize Roman text and whitespace; preserve meaningful address separators |
| `preprocessing/transliteration.py` | Convert supported Indic runs to one Roman representation; fold Latin accents afterward |
| `preprocessing/business_name.py` | Remove recognized boundary legal forms from the single processed business name |
| `preprocessing/address.py` | Route to country adapters and expose consistent address-component names |
| `preprocessing/us.py` | US-specific address parsing and normalization |
| `preprocessing/india.py` | Indian address extraction using lexical dictionaries and contextual rules |
| `preprocessing/france.py` | French address normalization and optional component extraction |
| `preprocessing/generic.py` | Fallback processing for other country labels |

Keep originals only in the unchanged input TSVs. Processed batches replace `business_name` and `business_address` with one Roman-script representation; do not append original-script, full-name or legal-form copies. Perform minimal Unicode cleanup before transliteration and accent folding afterward. Collapse whitespace without joining words. French uses accent folding rather than Indic script conversion.

The enriched record should retain normalized whole-name and whole-address text plus available components: postal code, house/plot/flat number, floor, building, locality/society, town/village/city, state, and landmark/direction. These are optional extracted fields, not mandatory inputs. Country adapters should return consistent component names, leave unavailable components null, and preserve unassigned address text. Incomplete parsing must not prevent matching.

### Lexical resources

| File | Intended contents |
| --- | --- |
| `resources/legal_suffixes.json` | Versioned normalized boundary-removal phrases and country rules |
| `resources/address_aliases.json` | Address abbreviations and normalization mappings |
| `resources/city_aliases.json` | City spelling variants and aliases |

`legal_suffixes.json` now contains versioned boundary-removal rules; address aliases now hold versioned US street-type/direction mappings; the standalone city alias resource contains the user-supplied US/French entries; the India address lexicon contains a separate finite list of common Indian city aliases. Define schemas when implementing additional consumers. Keep dictionary content outside processing functions so it can be reviewed independently. Unknown terms must continue through generic text processing.

### Blocking, learning, and matching

| File | Intended responsibility |
| --- | --- |
| `blocking.py` | Retrieve plausible S1-to-S2/S3 pairs through complementary routes and deduplicate pairs |
| `features.py` | Compute shared pairwise features for training and inference |
| `labels.py` | Expand ground-truth lists into positive pairs and label retrieved candidates |
| `splitting.py` | Partition by reference entity, keeping its labeled matching records together |
| `training.py` | Fit the pair classifier and save model, fitted feature state, and selected threshold |
| `evaluation.py` | Measure candidate recall and final matching quality; select a validation threshold |
| `inference.py` | Load fitted state, score candidate pairs in batches, and select zero-to-many matches |
| `submission.py` | Validate and export final matching pairs and the submission TSV |

Blocking selects plausible pairs for comparison. The classifier determines whether each candidate pair represents the same business. Sharing a block is not itself a positive label.

Candidate routes can include postal-code/name rules, city aliases, and character n-gram name retrieval. Union their results so incorrect or missing location information does not automatically eliminate a true match. Store pair IDs and retrieval indicators rather than duplicating full records for every candidate.

Features can include Jaccard similarity, edit similarity, TF-IDF cosine, component agreement, numeric conflicts, and missing-value indicators. IDs are for joins and output, not predictive features. Two absent components must not count as positive agreement.

Use difficult nonmatching candidates as training negatives when the ground truth exhaustively lists matches. LightGBM is the selected first classifier; it is not implemented yet. XGBoost is an optional later benchmark.

## End-to-end workflow

```text
Source TSVs
    → input validation
    → minimal cleanup, conditional transliteration, Roman normalization and legal-form removal
    → address-component extraction (US/French/Indian/generic)
    → candidate blocking
    → pairwise features
    → trained classifier scores
    → threshold-based match selection
    → final matching pairs and submission TSV
```

The intended training and prediction sequence is:

1. Create an entity-level training/validation split before fitting learned preprocessing or feature state. Keep each reference entity and its labeled matching records in the same partition.
2. Preprocess records and retrieve candidates separately for each partition.
3. Measure candidate recall: the fraction of true pairs retained by retrieval.
4. Compute features, attach ground-truth labels, and fit the classifier on training pairs.
5. Score the full validation candidate set, select a threshold using the official metric, and evaluate complete match lists. Count true matches missed by blocking when measuring final recall, and include entities with no matches.
6. Refit on all training data using the selected settings and save the final model bundle.
7. Process test data, retrieve candidates, and score pairs using the saved feature state and model.
8. Write and validate final outputs for every test Source 1 entity.

Persist fitted vectorizers, feature-column order, model, threshold, configuration, and resource versions or hashes together. Prediction should reload fitted feature state rather than independently fitting a different representation. Use batch processing and saved intermediates to manage large inputs; avoid exhaustive cross-source Cartesian comparisons.

## Intermediate table contracts

| Stage | Row represents | Expected information |
| --- | --- | --- |
| Preprocessing | One source record | ID, source, country, single Roman name/address, optional components |
| Blocking | One candidate pair | `source1_entity_id`, `candidate_entity_id`, retrieval indicators |
| Features | One candidate pair | Pair IDs and numeric comparison features |
| Labeling | One training candidate pair | Pair IDs, features, binary label |
| Scoring | One candidate pair | Pair IDs, match score, selection flag |
| Submission | One test Source 1 entity | `source1_entity_id`, `matched_entity_ids` |

Preserve IDs and postal codes as strings, including leading zeros. Deduplicate candidate pairs before feature computation and matching IDs before export.

## Configuration and storage

The current [default configuration](src/configs/default.json) is:

```json
{
  "paths": {
    "train_data": "../../dataset/train",
    "test_data": "../../dataset/test",
    "artifacts": "artifacts",
    "final_output": "../../output"
  },
  "random_seed": 42
}
```

The implemented loader resolves these paths relative to `code/business_entity_resolution/`, independent of the shell working directory. Relative config filenames use the same base, even when the config file itself lives elsewhere. Absolute paths and `~` expansion are supported. Add retrieval limits, batch sizes, model parameters, and evaluation settings when those stages are implemented.

| Location | Contents |
| --- | --- |
| `dataset/` | Original challenge inputs |
| `code/business_entity_resolution/artifacts/` | Processed records, candidate pairs, features, model bundles, logs, validation predictions, metrics, and run configuration |
| `output/` | Final results and final matching pairs only |

Do not save models, candidate blocks, caches, configuration snapshots, or validation reports in `output/`. Intermediate serialization formats can be chosen during implementation. Final tabular outputs must be TSV.

## Setup and execution status

From the repository root, prepare an isolated Python environment:

```bash
python3 -m venv code/business_entity_resolution/.venv
source code/business_entity_resolution/.venv/bin/activate
cd code/business_entity_resolution
```

Configuration uses the standard library; batch tables use Polars 1.44.2 and script conversion uses indic-transliteration 2.3.82; US and Indian parsing use Lark 1.3.1; usaddress 0.5.16 remains available for comparison. Python 3.12 is the verified runtime. Install the pinned dependency with:

```bash
python -m pip install -r requirements.txt
```

The current entry point can be checked with:

```bash
python -m src
```

It intentionally exits with `Pipeline scaffold only: execution is not implemented yet.` This is not a training or prediction command.

The planned CLI below is an implementation target. **These commands are not supported yet and cannot reproduce results with the current scaffold.**

```bash
python -m src prepare --config src/configs/default.json
python -m src train --config src/configs/default.json
python -m src predict --config src/configs/default.json

# Planned wrapper for the complete sequence:
python -m src run --config src/configs/default.json
```

| Planned command | Work performed |
| --- | --- |
| `prepare` | Validate and enrich train/test records; save intermediates |
| `train` | Generate training candidates/features, validate, select threshold, and fit the final model |
| `predict` | Generate test candidates/features, score pairs, and export final results |
| `run` | Execute prepare → train → predict |

After implementation, replace this planned interface with verified commands and record dependency versions, runtime/memory requirements, and actual artifact filenames. End-to-end reproduction is not available until the processing stages are implemented.

## Final output contract

The proposed submission path is `output/submission.tsv`. It must contain `source1_entity_id` and `matched_entity_ids` separated by a literal tab. IDs within the second cell are comma-separated.

The writer must ensure:

- Every ID in `test_source1.tsv` appears exactly once, including France and unfamiliar country labels.
- Selected match IDs exist in test Source 2 or Source 3.
- Matching IDs are unique and deterministically ordered within each row.
- No-match entities have an empty second cell, not a textual `NaN` or `None`.
- Multiple matching records are retained; no top-one decision is forced.

An additional proposed final file, `output/matching_pairs.tsv`, can contain one accepted pair per row with columns `source1_entity_id` and `matched_entity_id`. This pair-file name and schema are project conventions to confirm against the organizers' exact deliverable requirements. It contains final accepted matches, not all retrieved candidates. Neither file has been generated yet.

## Implementation order

1. Implement configuration, input validation, and the output contract.
2. Add conservative preprocessing and complementary candidate retrieval.
3. Implement shared features, entity-level splitting, and ground-truth labels.
4. Add model fitting, retrieval evaluation, and threshold selection.
5. Connect batch inference and submission validation.
6. Improve dictionaries and parsing using measured errors, and record methodology and experiments in `Documentation_template.md`.

## Component progress

1. **Configuration — complete.** `load_config()` reads the default JSON or a
   custom file and returns immutable settings with absolute paths. It rejects
   missing/unknown settings, invalid path values, seeds outside `0..4294967295`
   (including booleans), and overlapping input/output or intermediate/final
   directories. Failures raise `ConfigError`. Loading performs no writes and
   does not require the configured directories to exist.
2. **TSV input and schema validation — complete.** See usage below.
3. **Roman text preprocessing and legal-form removal — complete.** See below.
4. **US/French/Indian/generic address adapters — implemented within the documented scope.** France and India use project-owned deterministic rules; libpostal was evaluated on a US sample and not adopted.
5. **Indian address adapter — implemented.** It uses bounded lexical spans, Lark component/assembly rules, six-digit PIN recognition, state aliases, and partial extraction.

From `code/business_entity_resolution/`, try the implemented component:

```bash
python3 -c 'from src.config import load_config; print(load_config())'
python3 -m unittest discover -s tests -v
```

The pipeline CLI remains a scaffold until orchestration is implemented.

## Read-only data loading with Polars

Activate the project virtual environment and install `requirements.txt` first.
From `code/business_entity_resolution/`:

```python
from src.config import load_config
from src.data_io import load_dataset, read_source

config = load_config()
train = load_dataset(config, "train", batch_size=100_000)
print(train.row_counts)
for batch in train.sources["S1"]:
    # Process this Polars DataFrame here; do not accumulate batches in a list.
    print(batch.height)

# Single-pass reading when full cross-file validation is not needed:
for batch in read_source(config.paths.test_data / "test_source1.tsv", "S1"):
    print(batch.height)
```

`read_source` and `read_ground_truth` return iterators of Polars DataFrames.
`load_dataset` fully validates a split, then returns re-iterable `BatchTable`
readers plus `row_counts`. Consuming these readers opens inputs again: validation
and processing are separate passes. Test has `ground_truth=None`. Ground-truth
batches contain string `source1_entity_id` and `List(String)` `matched_entity_ids`
columns; no-match records have empty lists. This replaces the previous Python
list-of-dictionaries and dictionary-of-tuples interfaces.

The standard-library CSV parser reads strict TSV rows into bounded column buffers;
Polars stores each batch. Retaining this parser preserves rejection of short/long
rows and blank records rather than filling missing fields. Quoted tabs/newlines,
UTF-8/BOM, leading zeros, extra columns, empty strings and unknown countries keep
their prior behavior. Original source text is preserved. Spaces around matching
IDs are stripped. IDs must be unique across the entire file, including batches.
Ground truth rejects duplicate references/matches, invalid IDs and empty list
items. Full training validation checks complete S1 coverage and existence of
all referenced S2/S3 IDs. Errors raise `DataValidationError` with file context;
structural row errors include line numbers.

Input files are opened only for reading. The reader writes no inputs, outputs,
cache files or directories. Batch size defaults to 100,000 and must be a positive
integer. Iterators validate as consumed: exhaust them for full-file validation,
or close them if stopping early. Header-only files yield no batches. Keep inputs
stable between validation and processing. Avoid collecting all batches or
converting entire files back to Python dictionaries.

Only a batch's text stays in memory, but exact duplicate and cross-file checks
retain ID indexes proportional to the number of records; memory is not constant.
Tests use tiny batches to check cross-batch duplicates and matches, as well as
preservation of source values and read-only file access.

Follow the [agreed technology stack](#agreed-technology-stack) for future stages.
Only Polars was added as a dependency in the data-loading step.

### Reader benchmark

A single local run on `train_source1.tsv` (2,206,821 records), Python 3.12 and
Polars 1.44.2, with each reader in a separate process:

| Reader | Elapsed time | Peak process memory |
| --- | --- | --- |
| Previous list of dictionaries | 5.17 seconds | 1,135 MiB |
| Polars batches of 100,000 | 4.16 seconds | 303 MiB |

This measured approximately 73% lower peak memory. The old reader ran first,
so filesystem cache effects may favor the new reader; these are indicative
single-run measurements, not a general speed guarantee. Both consumed and
validated the complete source. Full split validation still requires ID indexes.

## Batch text preprocessing

The agreed policy is one processed representation per text field, with originals
available only in the read-only input TSVs. No duplicate script columns or
extracted legal-form columns are added. `preprocess_batch` returns a new frame;
it does not mutate its input frame, IDs, country labels, extra columns or row order.
No processed files are written yet. Address-component extraction is a separate step after this text stage, documented below.

```python
from src.config import load_config
from src.data_io import read_source
from src.preprocessing import preprocess_batches

config = load_config()
reader = read_source(config.paths.train_data / "train_source2.tsv", "S2")
for batch in preprocess_batches(reader):
    # business_name/business_address now contain processed Roman text.
    # Process this batch without collecting the entire iterator in memory.
    print(batch.height)
```

The implemented order and rules are:

1. Normalize Unicode with NFKC (including full-width Latin characters), retaining
   Indic marks until conversion. Null text becomes an empty string.
2. Convert runs of Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Tamil, Telugu,
   Kannada and Malayalam using indic-transliteration's IAST scheme. Mixed
   Latin/Indic strings are supported. Decimal digits become ASCII digits.
   Explicit compatibility mappings cover observed modern characters the library
   leaves unsupported: Malayalam chillu consonants, Devanagari candra vowels
   and Odia WA. These are matching approximations, not reversible linguistic
   transcription. Other unsupported letters raise `UnsupportedScriptError`.
3. Case-fold converted text and fold Latin accents, including `œ → oe` and
   `æ → ae`; render IAST `ś`/`ṣ` as `sh`. Strip combining marks only after
   transliteration. Polars lowercases ASCII text and collapses whitespace.
4. In names, convert `&` to `and`, remove apostrophes and replace other
   punctuation with spaces. In addresses, keep `/`, `-`, `.`, commas, `#`,
   apostrophes, parentheses, `+` and `&`; normalize typographic dashes/quotes.
   Thus `12/3` and `12-14` retain their numeric structure.
5. Remove dictionary-listed legal forms only at name boundaries. Common rules
   cover Ltd, Pvt Ltd, LLC, LLP, Inc, Corp and variants. French forms such as
   SARL/SAS/SA apply only to configured French country labels. Initial Romanized
   Indian forms apply only to configured Indian labels. A name consisting only
   of legal forms retains a nonempty form. Ambiguous `co` and interior words
   are not removed. Dictionary coverage is intentionally finite.

`src/resources/legal_suffixes.json` uses `version: 1`, a `common` list of
normalized phrases, and `countries` groups with `labels` and `forms` lists.
Country labels select rules without changing the output country field. Rules
load once per process; restart after edits. Extend this reviewed resource when
measured errors justify it. Boundary removal can still remove meaningful words
(e.g. a name beginning with “Limited”); evaluate its effect on matching quality.

Python transliteration runs only on distinct non-ASCII values in each batch.
The replacement map is discarded between batches; no unbounded whole-dataset
text cache is retained. Polars performs native casing, punctuation, whitespace
and legal-boundary operations. Neither learned state nor ground-truth labels
are used by these deterministic transformations.

Deterministic transliteration does not translate words and may differ from
informal Roman spellings, especially loanwords and language-dependent vowels.
The current rules do not guarantee matching quality, comprehensive script
coverage or linguistic accuracy. This text stage does not expand address abbreviations or parse grammar; the
address adapters below provide separate extraction and limited US normalization. Known matches must later measure whether these
choices help retrieval and classification.

Implementation references: [Python Unicode normalization](https://docs.python.org/3/library/unicodedata.html),
[indic-transliteration](https://github.com/indic-transliteration/indic_transliteration_py),
and [Unicode Indic-script details](https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-12/).

Verification: all 33 component tests pass, and `pip check` reports no broken
requirements. A read-only smoke test processed the first 10,000 rows of each of
the six source TSVs (60,000 total), checking preserved IDs/schema/order and
idempotence. Full-file SHA-256 hashes and modification times were unchanged.
Sample batch transformations took 0.043–0.279 seconds per 10,000 rows in one local
run (excluding file loading/checksums). This is not full-dataset processing or a
matching-quality benchmark.

## Address-component extraction (US, France and generic fallback)

Run this step **after** Roman text preprocessing. The function appends shared
nullable string columns to each batch, preserving the whole `business_address`,
all other existing fields and row order. Calling it again replaces component
columns. It writes no input or output files. US, French, and Indian parsers all
use the shared nullable component schema; India has its own grammar and lexicon.

```python
from src.config import load_config
from src.data_io import read_source
from src.preprocessing import preprocess_batches
from src.preprocessing.address import enrich_address_batches

config = load_config()
raw = read_source(config.paths.test_data / "test_source1.tsv", "S1")
for batch in enrich_address_batches(preprocess_batches(raw)):
    print(batch.select("entity_id", "address_city", "address_parse_status"))
```

`parse_address(text, country)` returns an immutable
`AddressResult` for one processed address. `enrich_address_batch` and
`enrich_address_batches` expose the same contract for Polars batches. A per-batch
cache holds at most 4,096 parsed (text, country) pairs; no whole-dataset cache is
retained. Python parsing is sequential in this implementation. The US and India Lark
grammars and lexicons are compiled/loaded once per process and reused per record.

### Shared output columns

All component names below have an `address_` prefix in Polars output:

| Fields | Meaning |
| --- | --- |
| `house_number`, `street` | House/range identifier and street text |
| `unit`, `floor`, `building` | Optional within-building identifiers |
| `locality`, `city`, `state` | Optional location components |
| `postal_code`, `po_box` | String identifiers; retain leading zeros |
| `landmark_direction` | US relation plus landmark name, e.g. `near central park` |
| `unparsed_text` | Unassigned content; whole input address on generic/error fallback |
| `parser` | `us_regex_lark`, `india_regex_lark`, `fr_rules` or `generic`; explicit legacy comparison returns `usaddress` |
| `parse_status` | `parsed`, `partial`, `fallback` or `empty` |
| `parse_reason` | Optional fallback/partial reason, such as `incomplete_components` or `unmatched_text` |

Missing components are null, never fabricated. `parsed` is a structural result,
not a correctness score or proof of deliverability; even a model-assigned label
can be wrong. Street directions stay in the street field. Some reserved fields
remain null until their adapter can extract them reliably.

### Country behavior

- **US:** the default is the project-owned regex + Lark parser described below.
  It extracts independent spans before applying a compact address grammar.
  Partial results survive missing components, conflicting values and unfamiliar
  orderings. Unknown content remains in `unparsed_text`; no trained backend is
  called implicitly. Full state names map to postal codes; ZIP/ZIP+4 detection
  uses position and recognized components to avoid stealing numbered streets.
- **France:** default deterministic rules extract a leading house number with a
  recognized street marker, a unique five-digit postal-code token, and a restricted
  city tail. They support leading zeros, house ranges and bis/ter/quater. Multiple
  postcode candidates are left unassigned. CEDEX, building and other unrecognized
  text remains in `unparsed_text`. Rule-based extraction is always `partial` or
  `fallback`, not a validated postal address. Complex/incomplete French addresses
  may have few extracted components.
- **India:** project-owned regex + Lark extraction recognizes Indian number labels,
  building/locality markers, rural administrative terms, state aliases and six-digit
  PINs. The India lexicon includes a finite set of common city names and aliases; exact
  aliases are canonicalized when address and state/PIN context supports a city.
  Other cities can still be inferred from a short segment adjacent to that context;
  the alias list is not a complete gazetteer or address validation.
- **Other/missing countries:** generic fallback retains the whole processed
  address with `unsupported_country`. Empty addresses return `empty`.

Routing recognizes US/USA/United States, France/FR/FRA, and India/IN/IND aliases
case-insensitively. It never rewrites the stored country label.

### Parser selection decision

The project evaluated libpostal on a sample of US addresses. Its token coverage
was high, but several reordered or ambiguous examples were assigned incorrect
components, so coverage did not establish component accuracy. The dependency and
adapter were removed. Address parsing will use project-owned country-specific
logic, with whole-address and unparsed text retained for downstream matching.

See [usaddress documentation](https://github.com/datamade/usaddress) for the
separation between component parsing, normalization and address verification.

Historical adapter verification (before the custom US parser): all 45 tests passed and `pip check` reports no broken
requirements. A smoke test enriched the first 1,000 records from each source TSV
(6,000 total), preserving IDs, whole processed addresses, source-file SHA-256
hashes and modification times. Batch extraction took 0.051–0.077 seconds per
1,000 records in one local run, excluding input loading and text preprocessing.
These structural checks are not component-accuracy or matching-quality evaluation.

## Address information-retention audit (historical)

These results describe earlier adapters. Current US results are in
[the custom parser evaluation](#custom-us-regex--grammar-parser).

A later 27,000-record audit (18,000 US train/test; 9,000 France test only) found
no changed whole addresses or missing tested tokens/numeric groups when components
and unparsed text are combined. **This does not imply adequate extraction:**
14.1% of nonempty US addresses and all nonempty French samples needed unparsed
text to retain every tested token. French city extraction was only 31/8,824;
a five-digit house number can be incorrectly labeled as a postcode. Targeted
checks also exposed missing US reordered city/state fields and floor-as-unit
assignment. Keep whole-address features and treat these as unresolved parser
limitations before relying on component-based blocking.

All 45 existing unit tests pass; separate diagnostic expectations fail on 5 of 7
hand-specified cases. No production parsers were changed by this audit. See the
[report](artifacts/address_retention_audit.md) and
[reproducible audit script](tests/audit_address_retention.py) for scope, examples,
checksums and limitations. Original source files remained unchanged.

## Best-guess component normalization policy

Trust the parser's proposed component roles and normalize them consistently;
do not build exceptions for every possible interpretation. These are matching
features, not verified facts. The future classifier must weigh component
comparisons alongside whole-name/whole-address similarity, missingness and
parse status. Uncertain components must not become mandatory blocking conditions;
complementary retrieval must preserve candidates when parsed fields are wrong.

`preprocessing/address_normalization.py` loads version 1 of
`resources/address_aliases.json` once per process. The legacy US adapter uses
`us.street_types` and `us.directions`; French rules use `france.street_types`.
The custom US parser owns its aliases in `us_address_lexicon.json`. Restart
processes after changing resources.

- US: trust grammar-selected street suffixes and normalize them (`avenue` →
  `ave`, `road` → `rd`). Direction words in recognized street spans map to
  `n/s/e/w/ne/nw/se/sw`. Other name words stay available, including `st` in
  `St Charles Ave`. The legacy adapter continues to normalize its model labels.
- France: treat a recognized leading marker in an extracted road component as
  the street type. Normalize `av.`/`ave` to `avenue`, `bd` to `boulevard`, `r.` to
  `rue`, `rte` to `route`, and other dictionary-listed variants. French extraction recognizes the
  dictionary's abbreviated street markers after a house number.
- Unknown terms remain unchanged. No aliases are substituted across entire city,
  business-name or unparsed-text fields. Whole processed addresses are retained
  unchanged, and missingness/status/reason fields remain available.

Historical verification ran the then-current unit suite, including
ambiguous role labels, equivalent abbreviations, French dotted markers and
whole-text preservation. Backend-specific tests were removed with the libpostal
adapter; the suite was not rerun for this removal. The previous retention audit
describes the earlier parser version and was not rerun. Postcode/city extraction issues remain separate work.
The Indian address adapter is implemented; no classifier or blocking stage is implemented yet.

## User-supplied city dictionaries

`resources/city_aliases.json` now contains the supplied **50 US cities and 50
French communes**, with canonical names, aliases and US state codes. Business
counts/ranks are not used as matching features. Canonical output is lowercase,
accent-folded Roman text. Resource provenance is the user-provided lists, not
an independently verified gazetteer. Schema version 1 has `us`/`france` entry
lists (`city`, `aliases`, optional `state`/`requires_context`) and
`regional_aliases` for DFW/RDU. Dictionaries are loaded once per process.

`preprocessing/cities.py` supplies the city index to the custom US parser and
provides post-parse enrichment for the French and legacy US adapters. The
following enrichment behavior and measurements describe those older adapters;
custom US segmentation is documented below:

- Normalize exact known aliases in the parser's city field. Preserve unrecognized
  or conflicting existing city guesses, consistent with best-guess parsing.
- If city is missing, search complete comma/semicolon-delimited address segments,
  including supported postcode/state wrappers. Do not search arbitrary substrings
  of street names. Multiple distinct city matches remain unresolved.
- US: respect a supplied state code; short and regional aliases require matching
  state context. DFW/RDU remain best-guess mappings to the supplied cities, not
  proof that a metro-area address lies in that city. Do not invent a state from
  a city match. The custom US parser canonicalizes full state names using its own lexicon.
- France: accent/case/punctuation variants are recognized. Saint-Denis/Saint-Paul
  entries for Réunion require Réunion text or a `974xx` postcode before enrichment.
  Other user-supplied ambiguous aliases, such as Aix and Boulogne, remain guesses.
- Never apply US entries to France or French entries to US. Unknown countries
  and unsupported countries keep generic behavior. Preserve whole addresses, unparsed
  evidence and original parse-status/reason values; adding a city is not proof
  that an address is complete or correct.

All 56 tests pass. On the same seeded 27,000-row audit sample, US nonempty city
fields increased from 15,868 to **16,090 / 17,615** nonempty addresses; French city
fields increased from 31 to **3,873 / 8,824**. The French comparison also includes
the preceding abbreviation-normalization change. These are coverage counts, not
accuracy or model-quality measurements. Whole addresses and input-file hashes
were unchanged; no missing numeric groups were detected with remainder text.
The lexical audit flags 32 US rows after canonical city substitution; saved
examples show expected changes such as Saint Louis → St. Louis and Ft Worth →
Fort Worth. The subsequent audit rerun separately accounts for documented parser-city substitutions while retaining raw lexical flags.

The current detailed output is `artifacts/address_retention_audit.json`; the
older Markdown audit records the baseline and its unresolved structural issues.
Further PIN disambiguation and broader Indian city coverage remain evaluation work.

### Historical retention rerun after city enrichment

All 56 tests pass. The same 27,000-row sample showed **zero missing tested tokens
or numeric groups with components plus unparsed text**, after allowing known
city-field substitutions. The 32 raw US lexical flags are retained in the report
and explained by canonical city aliases. Whole processed addresses and source
file hashes/modification times remain unchanged. This measures retention, not
component accuracy: unparsed-only information and known assignment gaps remain.
See the [current rerun report](artifacts/address_retention_after_cities.md) for
component coverage, targeted failures, comparison methodology and limitations.

## Custom US regex + grammar parser

The default US adapter now follows the supplied shallow, prioritized component
parser design. France and India use their own deterministic country rules. It is a
matching-feature extractor, not geographic or postal-address verification.

| File | Role |
| --- | --- |
| `src/preprocessing/us.py` | Strong-span recognition, US context rules, component assembly and partial results |
| `src/preprocessing/component_grammar.py` | Offset-tracked tokens, non-overlapping span collection and cached Lark LALR parsing |
| `src/resources/us_address_lexicon.json` | Versioned unit/floor/building/street/locality/landmark vocabulary, aliases, all 50 states + DC and small example gazetteers |
| `src/resources/us_address.lark` | Independently parseable component productions and compact normal/reverse address grammar |
| `src/resources/city_aliases.json` | User-supplied US/French city names and aliases |
| `src/resources/india_address_lexicon.json` | Indian address terms, state aliases, and common city aliases |
| `src/preprocessing/us_legacy.py` | Preserved usaddress adapter for explicit evaluation; never an automatic fallback |


### Parsing process and grammar rules

The parser tokenizes the normalized address into lowercase tokens while keeping
their original character offsets. Regexes and the versioned US lexicon propose
bounded component spans (at most 16 tokens). Recognizers run in a deliberate
order: contextual state names/codes, units and PO boxes, floors, landmarks,
streets, ZIP codes, known cities, buildings/localities, then remaining city
candidates. Context checks help distinguish state abbreviations from ordinary
words and street suffixes, avoid reading a landmark's street as the delivery
street, and avoid treating a numbered street suffix as a ZIP. City aliases are
accepted only when the match is unique in the available state context; an
unknown city is considered only when a state and street/PO box have been
recognized: the candidate must be a short segment beside the state or the final
segment after a street.

Each proposed span is independently checked against its component production in
`us_address.lark`. Accepted spans cannot overlap, and a second value for an
already-filled field is retained as unparsed text rather than overwriting the
first. The final component sequence, ordered by its position in the input, is
then checked against the address assembly rules. Unclaimed token text is
removed from a copy of the original address and returned as `unparsed_text`;
separators split and trim the leftover chunks. Thus failed component or
assembly validation does not discard other independently recognized fields.

The grammar has two layers. Component rules describe the shape of a single
typed span:

| Component | Accepted shape |
| --- | --- |
| `unit` | One or more unit markers followed by an identifier, such as `Suite 4B` |
| `floor` | A floor key with a number/ordinal on either side, or a known special floor such as `basement` |
| `building`, `locality` | Words ending or beginning with the corresponding lexicon marker, or a known gazetteer phrase |
| `street` | Optional house number, street-name words, street-type key and optional direction; or a route/highway key followed by a number |
| `landmark` | A recognized relation/direction marker followed by landmark words |
| `po_box` | `PO Box` key followed by an identifier |
| `postal_code` | Optional postal label followed by a ZIP/ZIP+4 token |

Address assembly then allows a street with optional location, building and
within-building details, or a PO box; location order can include city, state,
locality and postal code. A landmark may precede or follow the core address.
The grammar also permits compact reverse arrangements with a city before the
street portion. It is intentionally shallow: it validates recognized component
shapes and supported orderings rather than trying to interpret every word or
enumerate whole-address regexes. The separate regex/lexicon recognizers decide
which words can serve as typed keys, identifiers, states, ZIPs and aliases.

The recognizers reserve contextual states, units and floors, protect landmark
phrases, then extract streets and contextual postal codes. A marked landmark
such as `near Main Street` is protected so its street is not mistaken for the
record's delivery street. Cities, buildings and localities use the remaining
spans. Known cities can follow a marked building without a comma; unknown
cities require a short segment next to a state or a final segment after a
street with explicit state context. A city alone never supplies a missing state.

Lark validates typed components and their compact normal/reverse assembly;
it does not try hundreds of whole-address regex permutations. Name candidates
are bounded to 16 tokens per component; unknown city candidates to five.
Street directions stay in `street`, and landmark relation/name share the existing
`landmark_direction` field, preserving the batch schema. Unit IDs, house ranges,
alphanumeric house numbers, PO boxes and ZIP+4 retain meaningful digits/zeros.
Floor ordinals canonicalize to `floor N`; full state names become two-letter
codes. The small building/locality gazetteers contain examples from the design,
not a comprehensive geographic database.

US postal codes are represented by the shared `postal_code` field. The parser
accepts bare five-digit ZIP and ZIP+4 values in address context, plus explicit
`zip`, `zip code`, `zipcode`, `postal code`, `pin code` and `pincode` markers.
The marker is consumed while the exact code, including leading zeros, is kept.

`parsed` requires a house/street or PO box, city and state, recognized assembly,
and no unmatched text. `partial` keeps available fields even when assembly
fails. Conflicting/repeated components remain in `unparsed_text` instead of
silently overwriting an earlier value. `parse_reason` exposes incomplete
components, unfamiliar order, conflicts and unmatched text. A structurally
complete parse can still contain an incorrect best guess.

```python
from src.preprocessing.address import parse_address

result = parse_address("500 Park Avenue Suite 400 New York NY", "US")
# house_number='500', street='park ave', unit='suite 400',
# city='new york city', state='ny', parser='us_regex_lark'

result = parse_address("Suite 420, Jefferson Tower, Somewhere near Central Park", "US")
# unit='suite 420', building='jefferson tower',
# landmark_direction='near central park', unparsed_text='Somewhere'
```

`enrich_address_batches(preprocess_batches(read_source(...)))` continues to work
unchanged and preserves whole processed addresses and input frames. The parser
opens only its resources for reading and writes no files. Original TSVs stay
read-only. Lark 1.3.1 is installed in `.venv` and pinned in `requirements.txt`.

### US grammar validation and limits

Run from `code/business_entity_resolution/`:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
.venv/bin/python tests/evaluate_us_grammar.py --per-source 10000 \
  --output artifacts/us_grammar_coverage_60000.json
```

The current evaluation is a deterministic **60,000-row US sample**, with 10,000
rows selected from each train/test source using seeded entity-ID hashes. It
contains 58,683 nonempty addresses. The script compares the custom parser with
the preserved usaddress adapter and verifies source-file hashes/modification
times. The parser-only comparison is saved as
[`artifacts/us_grammar_coverage_60000.json`](artifacts/us_grammar_coverage_60000.json).
A second run applied the real `preprocess_batch → parse_address` order and saved
[`artifacts/us_pipeline_coverage_60000.json`](artifacts/us_pipeline_coverage_60000.json).
Normalization produced the same status and component counts on this sample.

All **68 unit tests pass**, including the specification examples, all state
names/codes, PIN/ZIP labels, partial/conflicting components, unchanged batch
inputs and existing French integration checks. `pip check` reports no broken
requirements. The 24 manually inspected development cases match all four expected
core fields in **24/24** custom parses versus **21/24** legacy parses. Those cases
were used during development and are not an independent accuracy estimate.

| Measurement on 58,683 nonempty addresses | Custom regex + Lark | Legacy usaddress |
| --- | ---: | ---: |
| Structurally parsed | 42,741 (72.83%) | 46,274 (78.85%) |
| Partial | 15,942 (27.17%) | 11,694 (19.93%) |
| Fallback | 0 | 715 (1.22%) |
| Nonempty house-number fields | 51,170 (87.20%) | 54,921 (93.59%) |
| Nonempty street fields | 53,434 (91.06%) | 56,552 (96.37%) |
| Nonempty city fields | 50,685 (86.37%) | 53,582 (91.31%) |
| Nonempty state fields | 58,683 (100%) | 53,650 (91.42%) |
| Nonempty floor fields | 547 | 0 |
| Addresses with unparsed text | 10,522 (17.93%) | 8,114 (13.83%) |
| Missing tested numeric groups, including remainder | 0 | 0 |
| Approximate rows/second, one warm-parser run | 4,907 | 7,081 |

For the custom parser, 26,297 of 344,081 sampled word/number tokens remained in
`unparsed_text` (7.64%). Another 5,420 rows were partial without leftover text:
they had recognized components but lacked the complete core fields or supported
assembly required for `parsed`. Frequent leftovers were numeric/rural-road forms,
`N/A`/`NULL`, repeated or unfamiliar unit/building markers, duplicated `#`, and
unknown/reordered location names. Examples include `290` from
`5157 County Road 290, Joplin, MO`, `Unit UNIT 12F3` from a complete street/city
address, and `Hammond` from `Hammond, 1704 Atchison Avenue, IN`.

Both parsers retained all tested numeric groups when component fields and
unparsed text were combined. Batch input columns, all six source SHA-256 hashes
and modification times were unchanged. Coverage is not accuracy: the sampled
rows have no component labels, and the custom parser fills fewer house/street/city
fields while retaining more uncertain text. Its state/floor coverage is higher.
A sampled `Fl 0, KS, ...` record also exposed a floor-versus-Florida ambiguity.
The comparison timing excludes file scanning, text normalization and batch
assembly. The production-order audit includes preprocessing and confirms the
same coverage figures; its token-remainder denominator is 344,079 normalized
word/number tokens. Both reports are sampled diagnostics rather than full-data
accuracy evaluations.


Known limitations include streets without listed types, unnamed/numeric-only
roads, rural-route conventions, ambiguous locality-versus-city names, finite
city dictionaries, and multiple addresses in one record. A locality marker can
still capture a name that should be a city. Only one value per component is
returned; later conflicts stay as text. Whole-address features and unparsed text
remain necessary for downstream matching. Neither matching quality nor full-
dataset accuracy has been measured, and a lexicon alone does not provide French or Indian ordering/context rules.


## Indian regex + grammar parser

The India route (`India`, `IN`, `IND`) uses offset-preserving token spans and
`resources/india_address.lark`, with aliases from `resources/india_address_lexicon.json`.
Location anchors are resolved from the large-scale end, then explicit delivery
components and bounded localities are extracted. Assembly applies a strong
**house/unit/floor/building → street → locality → city → administrative area → state → PIN**
prior. Residential components can exchange positions; landmarks do not affect
assembly order. Explicit fields in reverse addresses are retained as partial parses.

India-only normalization converts valid Roman numerals next to floor/phase/sector
markers (`III Floor` → `3 floor`), leaving names and initials intact. It also
separates PINs attached to known locations and joins spaced PINs with location or
postal-label context. Explicit number markers, bare leading IDs and slash IDs,
locality markers at either end, and multiword/suffix city aliases are supported.
City extraction no longer requires a previously recognized delivery component.
Multiple locality spans are retained in source order. Unknown and conflicting
text remains available for downstream matching.

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

Detailed results: [current JSON](artifacts/india_pipeline_coverage_60000.json), [saved baseline](artifacts/india_pipeline_coverage_60000_before.json), [audit script](tests/audit_india_pipeline_60000.py), [India regression tests](tests/test_india.py).
