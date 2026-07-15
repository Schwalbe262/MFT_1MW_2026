import json
from pathlib import Path
import tempfile
import unittest

from regression_260707.pipeline.artifacts import GenerationStore


class GenerationStoreTests(unittest.TestCase):
    def test_publish_is_content_addressed_idempotent_and_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            source.write_text("stable payload", encoding="utf-8")
            store = GenerationStore(root / "store")
            first = store.publish_files(
                "dataset", {"nested/data.txt": source},
                metadata={"strict_full_rows": 4000},
                parents=["solver:a"],
            )
            second = store.publish_files(
                "dataset", {"nested/data.txt": source},
                metadata={"strict_full_rows": 4000},
                parents=["solver:a"],
            )

            self.assertEqual(first.generation_id, second.generation_id)
            self.assertEqual(first.path, second.path)
            self.assertTrue((first.path / "COMPLETED").is_file())
            self.assertEqual(
                (first.path / "nested" / "data.txt").read_text(encoding="utf-8"),
                "stable payload",
            )
            self.assertEqual(store.load(first.path).manifest, first.manifest)

    def test_tamper_and_incomplete_generation_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data.bin"
            source.write_bytes(b"abc")
            store = GenerationStore(root / "store")
            generation = store.publish_files("model", {"data.bin": source})
            (generation.path / "data.bin").write_bytes(b"abd")
            with self.assertRaisesRegex(RuntimeError, "artifact mismatch"):
                store.load(generation.path)

            incomplete = root / "store" / "model" / ("0" * 64)
            incomplete.mkdir()
            (incomplete / "manifest.json").write_text(
                json.dumps({}), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                store.load(incomplete)

    def test_tree_publication_excludes_nested_generation_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tree = root / "tree"
            tree.mkdir()
            (tree / "result.json").write_text("{}", encoding="utf-8")
            (tree / "COMPLETED").write_text("old", encoding="utf-8")
            generation = GenerationStore(root / "store").publish_tree(
                "optimization", tree
            )
            self.assertEqual(
                sorted(generation.manifest["artifacts"]), ["result.json"]
            )


if __name__ == "__main__":
    unittest.main()
