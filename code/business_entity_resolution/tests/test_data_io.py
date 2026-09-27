"""TSV parsing, validation and read-only contract tests."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.config import PathsConfig, PipelineConfig
from src.data_io import DataValidationError, load_dataset, read_ground_truth as batches_truth, read_source as batches_source
import polars as pl


def read_source(path, source):
    return [row for batch in batches_source(path, source, batch_size=1) for row in batch.to_dicts()]


def read_ground_truth(path):
    return {row["source1_entity_id"]: tuple(row["matched_entity_ids"])
            for batch in batches_truth(path, batch_size=1) for row in batch.to_dicts()}

HEADER = 'entity_id\tbusiness_name\tbusiness_address\tcountry\n'
TRUTH = 'source1_entity_id\tmatched_entity_ids\n'


class DataIOTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.path = self.root / 'input.tsv'
        self.config = PipelineConfig(PathsConfig(self.root, self.root, self.root / 'a', self.root / 'o'), 42)

    def write(self, content):
        self.path.write_text(content, encoding='utf-8')
        return self.path

    def fixture(self, split='train'):
        for number in (1, 2, 3):
            (self.root / f'{split}_source{number}.tsv').write_text(
                HEADER + f'S{number}-001\tName\tAddress\tUnknown\n')
        if split == 'train':
            (self.root / 'train_ground_truth.tsv').write_text(TRUTH + 'S1-001\tS2-001,S3-001\n')

    def test_preserves_text_and_extra_fields(self):
        path = self.write('\ufeff' + HEADER.rstrip('\n') + '\tpostal\r\n'
                          'S1-001\t"Café\tभारत"\t"12, Road\nFloor 2"\tNew Country\t00123\r\n')
        row = read_source(path, 'S1')[0]
        self.assertEqual(row['business_name'], 'Café\tभारत')
        self.assertEqual(row['business_address'], '12, Road\nFloor 2')
        self.assertEqual(row['postal'], '00123')
        self.assertEqual(row['country'], 'New Country')

    def test_empty_text_and_header_only(self):
        self.assertEqual(read_source(self.write(HEADER), 'S1'), [])
        self.assertEqual(read_source(self.write(HEADER + 'S1-001\t\t\t\n'), 'S1')[0]['country'], '')

    def test_bad_source_files(self):
        for content in ('', 'entity_id\n', HEADER.replace('country', 'entity_id'),
                        HEADER + '\n', HEADER + 'S1-1\tx\ty\n',
                        HEADER + 'S1-1\tx\ty\tz\textra\n',
                        HEADER + 'S2-1\tx\ty\tz\n', HEADER + '\tx\ty\tz\n',
                        HEADER + 'S1-\tx\ty\tz\n', HEADER + 'S1- 1\tx\ty\tz\n',
                        HEADER + 'S1-1\tx\ty\tz\nS1-1\tx\ty\tz\n',
                        HEADER + 'S1-1\t"unterminated'):
            with self.subTest(content=content), self.assertRaises(DataValidationError):
                read_source(self.write(content), 'S1')

    def test_missing_and_invalid_encoding(self):
        with self.assertRaises(DataValidationError):
            read_source(self.path, 'S1')
        self.path.write_bytes(b'\xff')
        with self.assertRaises(DataValidationError):
            read_source(self.path, 'S1')

    def test_ground_truth_lists(self):
        result = read_ground_truth(self.write(TRUTH + 'S1-001\tS2-002, S3-003\nS1-004\t\n'))
        self.assertEqual(result, {'S1-001': ('S2-002', 'S3-003'), 'S1-004': ()})

    def test_bad_ground_truth(self):
        for rows in ('S1-1\tS2-1,S2-1\n', 'S1-1\tS2-1,\n', 'S1-1\tS1-2\n',
                     'S1-1\t\nS1-1\t\n', 'S2-1\t\n'):
            with self.subTest(rows=rows), self.assertRaises(DataValidationError):
                read_ground_truth(self.write(TRUTH + rows))

    def test_train_integrity(self):
        self.fixture()
        dataset = load_dataset(self.config, 'train', batch_size=1)
        self.assertEqual(next(iter(dataset.ground_truth))['matched_entity_ids'].to_list(), [['S2-001', 'S3-001']])
        for rows in ('S1-001\tS2-999\n', '', 'S1-999\t\n'):
            (self.root / 'train_ground_truth.tsv').write_text(TRUTH + rows)
            with self.subTest(rows=rows), self.assertRaises(DataValidationError):
                load_dataset(self.config, 'train')

    def test_test_split_needs_no_labels(self):
        self.fixture('test')
        self.assertIsNone(load_dataset(self.config, 'test').ground_truth)

    def test_read_only(self):
        self.fixture()
        def snapshot():
            return {p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
                    for p in self.root.iterdir()}
        before = snapshot()
        original = Path.open
        def read_only(path, mode='r', *args, **kwargs):
            self.assertEqual(mode, 'r')
            return original(path, mode, *args, **kwargs)
        with patch.object(Path, 'open', read_only):
            load_dataset(self.config, 'train')
        self.assertEqual(before, snapshot())

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            read_source(self.path, 'S4')
        with self.assertRaises(ValueError):
            load_dataset(self.config, 'validation')


    def test_polars_batches_and_repeatable_dataset(self):
        self.fixture()
        dataset = load_dataset(self.config, 'train', batch_size=1)
        self.assertEqual(dataset.row_counts['S1'], 1)
        for _ in range(2):
            batch = next(iter(dataset.sources['S1']))
            self.assertIsInstance(batch, pl.DataFrame)
            self.assertEqual(batch.schema['entity_id'], pl.String)
        path = self.write(HEADER + ''.join(f'S1-{n}\tName\tAddress\tX\n' for n in range(5)))
        batches = list(batches_source(path, 'S1', batch_size=2))
        self.assertEqual([b.height for b in batches], [2, 2, 1])
        self.assertEqual(pl.concat(batches)['entity_id'].to_list(), [f'S1-{n}' for n in range(5)])

    def test_invalid_batch_sizes(self):
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                batches_source(self.path, 'S1', batch_size=size)

    def test_cross_batch_matches(self):
        self.fixture()
        (self.root / 'train_source2.tsv').write_text(HEADER + 'S2-001\tA\tB\tX\nS2-002\tA\tB\tX\n')
        (self.root / 'train_ground_truth.tsv').write_text(TRUTH + 'S1-001\tS2-002\n')
        self.assertEqual(load_dataset(self.config, 'train', batch_size=1).row_counts['S2'], 2)
