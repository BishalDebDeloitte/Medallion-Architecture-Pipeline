"""Offline tests for the memory store (no embedding model may be loaded or downloaded)."""

import unittest
from pathlib import Path
from unittest.mock import patch

from core import memory
from core.paths import CHROMA_DIR


class MemoryTests(unittest.TestCase):
    def test_store_get_update_and_list(self):
        self.assertTrue(memory.store_document("mem-doc-1", "first text", {"run_id": "r1", "type": "intent"}))
        found = memory.get_document("mem-doc-1")
        self.assertEqual((found["document"], found["metadata"]["run_id"]), ("first text", "r1"))

        memory.store_document("mem-doc-1", "updated text", {"run_id": "r1", "type": "intent"})
        self.assertEqual(memory.get_document("mem-doc-1")["document"], "updated text")

        memory.store_document("mem-doc-2", "second", {"run_id": "r2"})
        ids = {item["id"] for item in memory.query_memory("anything", n_results=50)}
        self.assertTrue({"mem-doc-1", "mem-doc-2"} <= ids)

    def test_missing_document_is_none(self):
        self.assertIsNone(memory.get_document("does-not-exist"))

    def test_storage_goes_to_the_redirected_runtime_dir(self):
        memory.store_document("mem-doc-3", "x", {"k": "v"})
        self.assertTrue(CHROMA_DIR.exists())
        self.assertNotEqual(CHROMA_DIR.parent, Path(__file__).resolve().parent.parent)

    def test_no_embedding_model_is_requested(self):
        # The placeholder embedding is supplied explicitly, so Chroma never builds its default
        # (network-downloading) embedding function.
        with patch("chromadb.utils.embedding_functions.DefaultEmbeddingFunction") as default:
            memory.store_document("mem-doc-4", "y", {"k": "v"})
        default.assert_not_called()

    def test_failures_return_false_instead_of_raising(self):
        with patch.object(memory, "get_collection", side_effect=RuntimeError("disk full")):
            self.assertFalse(memory.store_document("mem-doc-5", "z"))
            self.assertIsNone(memory.get_document("mem-doc-5"))
            self.assertEqual(memory.query_memory("q"), [])


if __name__ == "__main__":
    unittest.main()
