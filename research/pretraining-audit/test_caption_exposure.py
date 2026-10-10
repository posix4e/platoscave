"""Synthetic checks of attribution ambiguity, source splits, and normalization."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from check_caption_exposure import audit


class ExposureAuditTest(unittest.TestCase):
    def run_fixture(self, images, groups):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "captions.zip"
            with zipfile.ZipFile(reference, "w") as archive:
                archive.writestr("dataset_coco.json", json.dumps({"images": images}))
            corpus = root / "groups.jsonl.gz"
            with gzip.open(corpus, "wt", encoding="utf-8") as stream:
                for group in groups:
                    stream.write(json.dumps(group) + "\n")
            return audit(corpus, reference)

    @staticmethod
    def image(source_id, original_split, captions, karpathy_split="train"):
        return {"cocoid": source_id, "filepath": original_split,
                "filename": f"COCO_{original_split}_{source_id:012d}.jpg",
                "split": karpathy_split,
                "sentences": [{"raw": text} for text in captions]}

    def test_original_source_split_overrides_karpathy_split(self):
        result = self.run_fixture([
            self.image(1, "val2014", ["first", "second"], "train"),
            self.image(2, "train2014", ["third", "fourth"], "test"),
        ], [{"guid": "1", "texts": ["first", "second"]},
            {"guid": "2", "texts": ["third", "fourth"]}])
        exact = result["modes"]["exact"]
        self.assertEqual(exact["unique_candidate_rows_by_original_split"], {"val2014": 1, "train2014": 1})
        self.assertEqual(exact["matching_source_images"]["unique_whole_group_candidate"]["source_ids_by_original_split"]["val2014"], [1])
        self.assertEqual(exact["guid_diagnostic"]["known_source_all_captions_match_rows_by_original_split"], {"val2014": 1, "train2014": 1})

    def test_ambiguous_generic_and_incompatible_or_missing_caption(self):
        result = self.run_fixture([
            self.image(1, "train2014", ["generic", "shared", "only-one"]),
            self.image(2, "val2014", ["generic", "shared", "only-two"]),
        ], [{"guid": "1", "texts": ["generic", "shared"]},
            {"guid": "not-numeric", "texts": ["generic", "generic"]},
            {"guid": "1", "texts": ["only-one", "only-two"]},
            {"guid": "999", "texts": ["only-one", "absent"]}])
        exact = result["modes"]["exact"]
        self.assertEqual(exact["whole_group_candidate_count_histogram"], {0: 2, 2: 2})
        self.assertEqual(exact["rows_with_all_texts_matching_somewhere"], 3)
        self.assertEqual(exact["rows_with_only_one_distinct_text"], 1)
        self.assertEqual(exact["matching_source_images"]["unique_whole_group_candidate"]["count"], 0)
        self.assertEqual(exact["guid_diagnostic"]["row_counts_by_status"], {
            "known_source_all_captions_match": 1, "known_source_caption_mismatch": 1,
            "nonnumeric": 1, "numeric_source_absent_from_reference": 1})

    def test_normalization_is_separate_and_retains_punctuation(self):
        result = self.run_fixture([
            self.image(1, "val2014", ["Ａ cat", "Straße dog."]),
        ], [{"guid": "1", "texts": ["a  cat", "STRASSE\tdog."]},
            ["a cat", "strasse dog"]])
        exact, normalized = result["modes"]["exact"], result["modes"]["normalized"]
        self.assertEqual(exact["matching_text_occurrences"], 0)
        self.assertEqual(normalized["matching_text_occurrences"], 3)
        self.assertEqual(normalized["whole_group_candidate_count_histogram"], {0: 1, 1: 1})
        self.assertEqual(normalized["guid_diagnostic"]["row_counts_by_status"], {"absent": 1, "known_source_all_captions_match": 1})
        self.assertEqual(exact["guid_diagnostic"]["row_counts_by_status"], {"absent": 1, "known_source_caption_mismatch": 1})


if __name__ == "__main__":
    unittest.main()
