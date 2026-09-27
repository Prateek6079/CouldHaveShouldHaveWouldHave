"""Run from the project directory with python3 -m unittest discover -s tests."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from src.config import ConfigError, DEFAULT_CONFIG, PROJECT_ROOT, load_config


class ConfigTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / "config.json"
        self.raw = json.loads(DEFAULT_CONFIG.read_text())

    def load_custom(self):
        self.path.write_text(json.dumps(self.raw))
        return load_config(self.path)

    def test_working_directory_independence(self):
        previous = Path.cwd()
        try:
            os.chdir(self.path.parent)
            config = load_config("src/configs/default.json")
            self.assertEqual(config, load_config())
            self.assertEqual(config.random_seed, 42)
            self.assertEqual(config.paths.train_data, PROJECT_ROOT.parent.parent / "dataset/train")
        finally:
            os.chdir(previous)

    def test_external_config_uses_project_root(self):
        self.assertEqual(self.load_custom(), load_config())

    def test_absolute_path_without_side_effects(self):
        target = self.path.parent / "new_artifacts"
        self.raw["paths"]["artifacts"] = str(target)
        self.assertEqual(self.load_custom().paths.artifacts, target)
        self.assertFalse(target.exists())

    def test_schema(self):
        for raw in ([], {}, {**self.raw, "typo": 1}, {**self.raw, "paths": []},
                    {**self.raw, "paths": {"train_data": "data"}}):
            with self.subTest(raw=raw):
                self.path.write_text(json.dumps(raw))
                with self.assertRaises(ConfigError):
                    load_config(self.path)

    def test_invalid_paths(self):
        for value in (None, 1, "", "  ", "bad" + chr(0)):
            with self.subTest(value=value):
                self.raw["paths"]["train_data"] = value
                with self.assertRaises(ConfigError):
                    self.load_custom()

    def test_invalid_seeds(self):
        for seed in (True, -1, 2**32, 1.5, "42", None):
            with self.subTest(seed=seed):
                self.raw["random_seed"] = seed
                with self.assertRaises(ConfigError):
                    self.load_custom()

    def test_output_overlap(self):
        for target in ("../../output", "../../output/cache", "../..", "../../dataset/train/cache"):
            with self.subTest(target=target):
                self.raw["paths"]["artifacts"] = target
                with self.assertRaisesRegex(ConfigError, "must not overlap"):
                    self.load_custom()

    def test_missing_and_malformed_files(self):
        with self.assertRaisesRegex(ConfigError, "Cannot load configuration"):
            load_config(self.path)
        self.path.write_text("{broken")
        with self.assertRaisesRegex(ConfigError, "Cannot load configuration"):
            load_config(self.path)
