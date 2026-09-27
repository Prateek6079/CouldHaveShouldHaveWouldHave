"""Strict, read-only TSV ingestion in Polars batches.

Iterators validate as consumed: exhaust them to validate the entire file.
Only ID indexes grow with file size; full source records are never accumulated.
"""
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import polars as pl

from .config import PipelineConfig

SOURCE_COLUMNS = {"entity_id", "business_name", "business_address", "country"}
TRUTH_COLUMNS = {"source1_entity_id", "matched_entity_ids"}
DEFAULT_BATCH_SIZE = 100_000


class DataValidationError(ValueError):
    """An input file is unreadable or violates the input contract."""


def _batch_size(value: int) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError("batch_size must be a positive integer")


def _batches(path: Path, required: set[str], batch_size: int) -> Iterator[pl.DataFrame]:
    # Retain csv's strict row-width/blank-record checks and quoted-field semantics.
    # Polars CSV readers may pad short rows; that would weaken our input contract.
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t", strict=True)
            header = next(reader, None)
            if not header:
                raise DataValidationError(f"{path}: missing header")
            if len(header) != len(set(header)) or any(not name.strip() for name in header):
                raise DataValidationError(f"{path}: duplicate or blank column names")
            missing = required - set(header)
            if missing:
                raise DataValidationError(f"{path}: missing required columns {sorted(missing)}")
            columns = [[] for _ in header]
            size = 0
            for cells in reader:
                if len(cells) != len(header):
                    raise DataValidationError(
                        f"{path}: line {reader.line_num}: expected {len(header)} fields, got {len(cells)}"
                    )
                for column, cell in zip(columns, cells):
                    column.append(cell)
                size += 1
                if size == batch_size:
                    frame = pl.DataFrame(dict(zip(header, columns)), schema={c: pl.String for c in header})
                    columns, size = [[] for _ in header], 0
                    yield frame
            if size:
                yield pl.DataFrame(dict(zip(header, columns)), schema={c: pl.String for c in header})
    except (OSError, UnicodeError, csv.Error, pl.exceptions.PolarsError) as exc:
        raise DataValidationError(f"{path}: cannot read TSV: {exc}") from exc


def _validate_id(value: str, sources: set[str], context: str) -> None:
    source, separator, suffix = value.partition("-")
    if source not in sources or not separator or not suffix or any(c.isspace() for c in value):
        raise DataValidationError(f"{context}: invalid ID {value!r}")


def read_source(path: str | Path, source: str, *, batch_size: int = DEFAULT_BATCH_SIZE) -> Iterator[pl.DataFrame]:
    """Return validated string-column batches in file order.

    Empty text, extra columns and unknown countries are preserved. Header-only
    files yield no batches. Duplicate checking retains IDs across all batches.
    Errors during reading are raised on iteration, not iterator construction.
    """
    if source not in {"S1", "S2", "S3"}:
        raise ValueError("source must be S1, S2 or S3")
    _batch_size(batch_size)
    return _source_batches(Path(path), source, batch_size)


def _source_batches(path: Path, source: str, batch_size: int) -> Iterator[pl.DataFrame]:
    seen = set()
    for frame in _batches(path, SOURCE_COLUMNS, batch_size):
        for entity_id in frame["entity_id"]:
            _validate_id(entity_id, {source}, str(path))
            if entity_id in seen:
                raise DataValidationError(f"{path}: duplicate entity_id {entity_id!r}")
            seen.add(entity_id)
        yield frame


def read_ground_truth(path: str | Path, *, batch_size: int = DEFAULT_BATCH_SIZE) -> Iterator[pl.DataFrame]:
    """Yield S1 string IDs and List(String) match columns, with [] for no matches.

    Whitespace around matching IDs is stripped. Duplicate references (including
    across batches), duplicate matches, empty list items and invalid IDs fail.
    """
    _batch_size(batch_size)
    return _truth_batches(Path(path), batch_size)


def _truth_batches(path: Path, batch_size: int) -> Iterator[pl.DataFrame]:
    seen = set()
    for frame in _batches(path, TRUTH_COLUMNS, batch_size):
        lists = []
        for reference, cell in frame.select("source1_entity_id", "matched_entity_ids").iter_rows():
            _validate_id(reference, {"S1"}, str(path))
            if reference in seen:
                raise DataValidationError(f"{path}: duplicate reference ID {reference!r}")
            seen.add(reference)
            ids = [value.strip() for value in cell.split(",")] if cell.strip() else []
            for value in ids:
                _validate_id(value, {"S2", "S3"}, str(path))
            if len(ids) != len(set(ids)):
                raise DataValidationError(f"{path}: duplicate matching IDs for {reference}")
            lists.append(ids)
        yield frame.with_columns(pl.Series("matched_entity_ids", lists, dtype=pl.List(pl.String)))


@dataclass(frozen=True)
class BatchTable:
    """Re-iterable input: every iteration opens the file anew, read-only."""
    path: Path
    source: str | None
    batch_size: int

    def __iter__(self) -> Iterator[pl.DataFrame]:
        if self.source is None:
            return read_ground_truth(self.path, batch_size=self.batch_size)
        return read_source(self.path, self.source, batch_size=self.batch_size)


@dataclass(frozen=True)
class Dataset:
    sources: dict[str, BatchTable]
    ground_truth: BatchTable | None
    row_counts: dict[str, int]


def load_dataset(config: PipelineConfig, split: str, *, batch_size: int = DEFAULT_BATCH_SIZE) -> Dataset:
    """Validate a full split, returning re-iterable batch readers and row counts.

    Validation scans files once; consuming the returned readers reads them again.
    Training checks complete S1 coverage and existence of every matching ID.
    Only ID indexes are retained during validation, and released before return.
    No input or output files are written. Inputs must remain stable while used.
    """
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    _batch_size(batch_size)
    directory = config.paths.train_data if split == "train" else config.paths.test_data
    sources = {f"S{n}": BatchTable(directory / f"{split}_source{n}.tsv", f"S{n}", batch_size)
               for n in (1, 2, 3)}
    indexes, counts = {}, {}
    for source, table in sources.items():
        counts[source] = 0
        ids = set()
        for batch in table:
            counts[source] += batch.height
            if split == "train":
                ids.update(batch["entity_id"])
        if split == "train":
            indexes[source] = ids
    truth = None
    if split == "train":
        truth = BatchTable(directory / "train_ground_truth.tsv", None, batch_size)
        counts["ground_truth"] = 0
        for batch in truth:
            counts["ground_truth"] += batch.height
            for reference, matches in batch.select("source1_entity_id", "matched_entity_ids").iter_rows():
                if reference not in indexes["S1"]:
                    raise DataValidationError(f"{truth.path}: unknown reference ID {reference!r}")
                indexes["S1"].remove(reference)
                for value in matches:
                    if value not in indexes[value[:2]]:
                        raise DataValidationError(f"{truth.path}: {reference}: unknown matching ID {value!r}")
        if indexes["S1"]:
            raise DataValidationError(f"{truth.path}: {len(indexes['S1'])} missing reference IDs")
    return Dataset(sources, truth, counts)
