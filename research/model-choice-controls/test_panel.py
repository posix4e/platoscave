"""Synthetic-only checks of the frozen six-pair assembly, selection and evaluation.

No photographs, private features, model weights or model inference are used. Real
artifact pins are patched only inside temporary fixtures; the public primary
numerical core is loaded at its actual frozen SHA256.
"""

import copy
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import random
import re
import tempfile
from types import SimpleNamespace
import unicodedata
import unittest
from unittest import mock

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("tested_model_choice_panel", HERE / "panel.py")
panel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(panel)
CORE_PATH = HERE.parent / "five-photo-pilot" / "solve.py"
core = panel.load_module(CORE_PATH, "test_panel_original_core", panel.CORE_SHA)
PAIRS = [f"{image}__{text}" for image in ("dino", "moco", "mae") for text in ("glove", "elmo")]
EDGES = list(itertools.combinations(range(5), 2))
PERMUTATIONS = list(itertools.permutations(range(5)))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scalar_distances(array, order):
    """Independent scalar replay; no production cosine or matrix multiplication."""
    rows = [[float(v) for v in array[index]] for index in order]
    lengths = [math.sqrt(math.fsum(v * v for v in row)) for row in rows]
    return [[0.0 if i == j else 1.0 - math.fsum(
        (a / lengths[i]) * (b / lengths[j]) for a, b in zip(rows[i], rows[j]))
             for j in range(5)] for i in range(5)]


def independent_scores(x, y):
    return {p: math.fsum((x[i][j] - y[p[i]][p[j]]) ** 2 for i, j in EDGES) / 10
            for p in PERMUTATIONS}


class PanelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic-panel-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.release = self.root / "release"
        self.release.mkdir()
        save(self.release / "protocol.json", {"protocol_id": panel.PROTOCOL_ID,
             "status": "frozen_before_additional_photo_features"})
        for filename in ("panel.py", "vision_features.py", "elmo_features.py", "vision-vendor-lock.json"):
            (self.release / filename).write_text("Synthetic release fixture only: " + filename + "\n")
        save(self.release / "release-lock.json", {"schema_version": 1, "files": {
            p.name: sha(p) for p in sorted(self.release.iterdir())}})
        self.rows = [{"candidate_id": value, "label_verbatim": label}
                     for value, label in zip(panel.IDS,
                         ["Synthetic ALPHA", "Synthetic beta", "Synthetic gamma", "Synthetic delta", "Synthetic epsilon"])]
        self.manifest = self.root / "synthetic-manifest.json"
        save(self.manifest, {"rows": self.rows})
        self.base = self.root / "synthetic-base.npz"
        self.arrays = self.make_arrays()
        np.savez_compressed(self.base, image=self.arrays["dino"], text=self.arrays["glove"],
                            sample_ids=np.asarray(panel.IDS))
        for name, value in (("ROOT", self.release), ("MANIFEST_SHA", sha(self.manifest)),
                            ("BASE_SHA", sha(self.base))):
            patcher = mock.patch.object(panel, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.provenance = panel.release()
        self.args = SimpleNamespace(manifest=self.manifest, base=self.base, out=self.root / "assembled")
        document = {"sample_ids": panel.IDS, "captions": [r["label_verbatim"] for r in self.rows]}
        self.caption_sha = hashlib.sha256((json.dumps(document, indent=2, ensure_ascii=False,
                                                       allow_nan=False) + "\n").encode()).hexdigest()
        self.token_rows = []
        for row in self.rows:
            tokens = re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+", unicodedata.normalize("NFKC", row["label_verbatim"]).lower())
            self.token_rows.append({"sample_id": row["candidate_id"], "tokens": tokens, "count": len(tokens)})
        for encoder in ("moco", "mae", "elmo"):
            path = self.root / (encoder + ".npz")
            meta = self.root / (encoder + ".json")
            setattr(self.args, encoder, path)
            setattr(self.args, encoder + "_metadata", meta)
            np.savez_compressed(path, features=self.arrays[encoder], sample_ids=np.asarray(panel.IDS))
            record = {**self.provenance, "encoder": encoder, "status": "complete",
                      "feature_sha256": sha(path), "weights_sha256": panel.WEIGHTS[encoder],
                      "sample_count": 5, "sample_ids": panel.IDS, "shape": list(self.arrays[encoder].shape),
                      "dtype": "float32", "batch_size": 1, "helper_sha256": panel.HELPER_SHA,
                      "source_sha256": self.provenance["source_hashes"]["elmo_features.py"],
                      "options_sha256": "86cb4c1a7e2f25e3867d06810e33ae27fbbd511ac77ca377be5fb7405a7085b1",
                      "captions_sha256": self.caption_sha, "tokens": self.token_rows,
                      "runtime": {"pillow": "12.3.0"}}
            save(meta, record)
        self.selection = self.root / "selection"
        self.primary = self.root / "synthetic-primary-evaluation.json"
        self.evaluation = self.root / "evaluation.json"

    def make_arrays(self, tied=False):
        rng = np.random.default_rng(1486)
        latent = rng.normal(size=(5, 8)).astype(np.float32)
        latent /= np.linalg.norm(latent, axis=1, keepdims=True)
        raw = {"dino": latent, "glove": latent[[2, 4, 1, 0, 3]],
               "moco": rng.normal(size=(5, 8)), "mae": rng.normal(size=(5, 8)),
               "elmo": rng.normal(size=(5, 8))}
        arrays = {}
        for name, dimensions in panel.DIMENSIONS.items():
            value = np.zeros((5, dimensions), dtype=np.float32)
            if tied:
                value[:, :5] = np.eye(5, dtype=np.float32)
            else:
                value[:, :8] = raw[name]
                value /= np.linalg.norm(value, axis=1, keepdims=True)
            arrays[name] = value
        return arrays

    def install_arrays(self, arrays):
        self.arrays = arrays
        np.savez_compressed(self.base, image=arrays["dino"], text=arrays["glove"], sample_ids=np.asarray(panel.IDS))
        panel.BASE_SHA = sha(self.base)
        for encoder in ("moco", "mae", "elmo"):
            self.rewrite_features(encoder, arrays[encoder])

    def rewrite_features(self, encoder, array, ids=None):
        path = getattr(self.args, encoder)
        np.savez_compressed(path, features=array, sample_ids=np.asarray(panel.IDS if ids is None else ids))
        self.edit_metadata(encoder, feature_sha256=sha(path))

    def edit_metadata(self, encoder_name, **changes):
        path = getattr(self.args, encoder_name + "_metadata")
        value = read(path)
        value.update(changes)
        save(path, value)

    def assemble_and_select(self):
        panel.assemble(self.args)
        panel.solve(SimpleNamespace(input=self.args.out / "blind-panel.json", core=CORE_PATH, out=self.selection))
        self.blind = read(self.args.out / "blind-panel.json")
        self.truth_path = self.args.out / "private-ground-truth.json"
        self.truth = read(self.truth_path)
        # The historical comparison record is synthetic. Its values use the real
        # primary core to mirror that interface; independent tests below replay
        # all six cells without calling either production scoring helper.
        baseline = core.enumerate_assignments({"schema_version": 1, "protocol_id": core.PROTOCOL_ID,
            "protocol_sha256": self.provenance["protocol_sha256"], "feature_sha256": panel.BASE_SHA,
            "image_ids": self.blind["image_ids"], "text_ids": self.blind["text_ids"],
            "image_distances": self.blind["image_distances"]["dino"],
            "text_distances": self.blind["text_distances"]["glove"]})
        actual = next(v for v in baseline if v["mapping"] == self.truth["mapping"])
        save(self.primary, {"true_mapping": self.truth["mapping"], "labels": self.truth["labels"],
            "image_labels": self.truth["image_labels"], "true_cost": actual["cost"],
            "true_assignment_rank_interval": [1 + sum(v["cost"] < actual["cost"] - panel.TOL for v in baseline),
                                               sum(v["cost"] <= actual["cost"] + panel.TOL for v in baseline)],
            "best_assignments_evaluated": [{**v, "correct_out_of_5": sum(v["mapping"][i] == t
                for i, t in self.truth["mapping"].items())} for v in baseline if v["optimal"]]})
        patcher = mock.patch.object(panel, "PRIMARY_EVALUATION_SHA", sha(self.primary))
        patcher.start()
        self.addCleanup(patcher.stop)

    def evaluate(self):
        panel.evaluate(SimpleNamespace(selection=self.selection, core=CORE_PATH, ground_truth=self.truth_path,
                                       primary_evaluation=self.primary, out=self.evaluation))
        return read(self.evaluation)

    def refresh_selection_receipt(self):
        save(self.selection / "checksums.json", {name: sha(self.selection / name) for name in
             ("blind-panel.json", "assignments.json", "selected.json")})

    def test_all_six_cells_replay_all_120_assignments_and_primary(self):
        before = self.base.read_bytes()
        self.assemble_and_select()
        outcome = self.evaluate()
        records = read(self.selection / "assignments.json")["assignments"]
        self.assertEqual(list(records), PAIRS)
        self.assertEqual(list(outcome["pairs"]), PAIRS)
        self.assertEqual(sum(map(len, records.values())), 720)
        self.assertEqual(self.base.read_bytes(), before)
        with np.load(self.args.out / "features.npz", allow_pickle=False) as archive:
            self.assertEqual(set(archive.files), set(panel.DIMENSIONS) | {"sample_ids"})
            for name, array in self.arrays.items():
                np.testing.assert_array_equal(archive[name], array)
        costs_by_pair = []
        for pair in PAIRS:
            image, text = pair.split("__")
            scores = independent_scores(self.blind["image_distances"][image], self.blind["text_distances"][text])
            minimum = min(scores.values())
            best = {p for p, cost in scores.items() if cost <= minimum + 1e-12}
            self.assertEqual({tuple(v["text_indices"]) for v in records[pair]}, set(PERMUTATIONS))
            for record in records[pair]:
                permutation = tuple(record["text_indices"])
                self.assertEqual(record["cost"], scores[permutation])
                self.assertEqual(record["mapping"], {f"I{i:02d}": f"T{j:02d}" for i, j in enumerate(permutation)})
                self.assertEqual(record["optimal"], permutation in best)
                xedges = [self.blind["image_distances"][image][i][j] for i, j in EDGES]
                yedges = [self.blind["text_distances"][text][permutation[i]][permutation[j]] for i, j in EDGES]
                self.assertAlmostEqual(record["distance_correlation"], float(np.corrcoef(xedges, yedges)[0, 1]), places=14)
            result = outcome["pairs"][pair]
            true_indices = tuple(int(self.truth["mapping"][f"I{i:02d}"][1:]) for i in range(5))
            true_cost = scores[true_indices]
            self.assertEqual(result["true_cost"], true_cost)
            self.assertEqual(result["true_assignment_rank_interval"],
                [1 + sum(v < true_cost - 1e-12 for v in scores.values()),
                 sum(v <= true_cost + 1e-12 for v in scores.values())])
            correct = [sum(a == b for a, b in zip(p, true_indices)) for p in best]
            self.assertEqual(result["mean_correct_out_of_5"], sum(correct) / len(correct))
            self.assertEqual(result["correct_out_of_5_range"], [min(correct), max(correct)])
            nonbest = [cost for p, cost in scores.items() if p not in best]
            self.assertEqual(result["gap_to_first_distinct_cost"], min(nonbest) - minimum if nonbest else None)
            costs_by_pair.append(tuple(scores.values()))
        self.assertEqual(len(set(costs_by_pair)), 6, "Fixture must distinguish all six model pairings")
        baseline = read(self.primary)
        self.assertEqual(outcome["pairs"]["dino__glove"]["best_assignments_evaluated"], baseline["best_assignments_evaluated"])
        self.assertGreater(outcome["pairs"]["dino__glove"]["true_assignment_rank_interval"][0], 1)
        self.assertTrue(outcome["primary_result_unchanged"])

    def test_separate_shuffles_and_scalar_cosine_preserve_item_identity(self):
        panel.assemble(self.args)
        blind = read(self.args.out / "blind-panel.json")
        truth = read(self.args.out / "private-ground-truth.json")
        image_order, text_order = list(range(5)), list(range(5))
        random.Random(20261010).shuffle(image_order)
        random.Random(20261011).shuffle(text_order)
        self.assertNotEqual(image_order, text_order)
        self.assertEqual(blind["image_ids"], [f"I{i:02d}" for i in range(5)])
        self.assertEqual(blind["text_ids"], [f"T{i:02d}" for i in range(5)])
        self.assertFalse({"labels", "image_labels", "mapping", "captions"} & set(blind))
        self.assertFalse(any(row["label_verbatim"] in json.dumps(blind) for row in self.rows))
        for name in panel.DIMENSIONS:
            modality, order = ("image", image_order) if name in panel.IMAGES else ("text", text_order)
            np.testing.assert_allclose(blind[modality + "_distances"][name], scalar_distances(self.arrays[name], order), atol=1e-15, rtol=0)
        for i, original in enumerate(image_order):
            tid = truth["mapping"][f"I{i:02d}"]
            self.assertEqual(tid, f"T{text_order.index(original):02d}")
            self.assertEqual(truth["image_labels"][f"I{i:02d}"], self.rows[original]["label_verbatim"])
            self.assertEqual(truth["labels"][tid], self.rows[original]["label_verbatim"])

    def test_all_tied_graphs_report_every_optimum_without_favorable_tie_break(self):
        self.install_arrays(self.make_arrays(tied=True))
        self.assemble_and_select()
        outcome = self.evaluate()
        for value in outcome["pairs"].values():
            self.assertEqual(value["tied_count"], 120)
            self.assertEqual(value["true_assignment_rank_interval"], [1, 120])
            self.assertEqual(value["correct_out_of_5_range"], [0, 5])
            self.assertEqual(value["mean_correct_out_of_5"], 1.0)
            self.assertIsNone(value["gap_to_first_distinct_cost"])
            self.assertTrue(all(v["distance_correlation"] is None for v in value["best_assignments"]))

    def test_changed_original_archive_rejected_before_output(self):
        with self.base.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "Original feature archive changed"):
            panel.assemble(self.args)
        self.assertFalse(self.args.out.exists())

    def test_changed_manifest_rejected_before_output(self):
        data = read(self.manifest)
        data["rows"][0]["label_verbatim"] = "Different synthetic label"
        save(self.manifest, data)
        with self.assertRaisesRegex(ValueError, "Changed five-photo manifest"):
            panel.assemble(self.args)
        self.assertFalse(self.args.out.exists())

    def test_changed_frozen_source_rejected(self):
        (self.release / "vision_features.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "Changed frozen source"):
            panel.assemble(self.args)

    def test_additional_feature_hash_mismatch_rejected(self):
        with self.args.moco.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "Feature/metadata hash mismatch"):
            panel.assemble(self.args)

    def test_additional_id_order_rejected_even_with_updated_file_hash(self):
        self.rewrite_features("elmo", self.arrays["elmo"], panel.IDS[::-1])
        with self.assertRaisesRegex(ValueError, "Additional features have wrong IDs"):
            panel.assemble(self.args)

    def test_invalid_feature_values_shapes_and_dtypes_rejected(self):
        wrong_norm = self.arrays["moco"].copy()
        wrong_norm[0] = 0
        nonfinite = self.arrays["moco"].copy()
        nonfinite[0, 0] = np.nan
        cases = [self.arrays["moco"].astype(np.float64), self.arrays["moco"][:, :-1], wrong_norm, nonfinite]
        for array in cases:
            with self.subTest(shape=array.shape, dtype=str(array.dtype), finite=bool(np.isfinite(array).all())):
                self.rewrite_features("moco", array)
                with self.assertRaisesRegex(ValueError, "Wrong moco feature|Invalid moco features"):
                    panel.assemble(self.args)
                self.assertFalse(self.args.out.exists())

    def test_incomplete_metadata_and_wrong_model_bytes_rejected(self):
        for changes, message in [({"status": "synthetic_readiness_passed"}, "Incomplete feature extraction"),
                                  ({"weights_sha256": "0" * 64}, "Feature checkpoint mismatch"),
                                  ({"sample_count": 4}, "Wrong feature sample count")]:
            with self.subTest(changes=changes):
                original = read(self.args.moco_metadata)
                self.edit_metadata("moco", **changes)
                with self.assertRaisesRegex(ValueError, message):
                    panel.assemble(self.args)
                save(self.args.moco_metadata, original)

    def test_elmo_caption_hash_ids_and_tokens_bound_to_locked_labels(self):
        original = read(self.args.elmo_metadata)
        altered_tokens = copy.deepcopy(original["tokens"])
        altered_tokens[0]["tokens"] = ["different"]
        cases = [({"captions_sha256": "0" * 64}, "ELMo caption input changed"),
                 ({"sample_ids": panel.IDS[::-1]}, "ELMo caption input changed"),
                 ({"tokens": altered_tokens}, "ELMo token record changed")]
        for changes, message in cases:
            with self.subTest(changes=changes):
                self.edit_metadata("elmo", **changes)
                with self.assertRaisesRegex(ValueError, message):
                    panel.assemble(self.args)
                save(self.args.elmo_metadata, original)

    def test_source_provenance_and_elmo_options_rejected(self):
        for encoder, changes, message in [
            ("moco", {"protocol_sha256": "0" * 64}, "Vision feature source mismatch"),
            ("elmo", {"source_sha256": "0" * 64}, "ELMo source mismatch"),
            ("elmo", {"options_sha256": "0" * 64}, "ELMo options mismatch")]:
            with self.subTest(encoder=encoder, changes=changes):
                path = getattr(self.args, encoder + "_metadata")
                original = read(path)
                self.edit_metadata(encoder, **changes)
                with self.assertRaisesRegex(ValueError, message):
                    panel.assemble(self.args)
                save(path, original)

    def test_vision_encoder_preprocessing_batch_and_pillow_recipe_rejected(self):
        original = read(self.args.moco_metadata)
        for change in ({"encoder": "mae"}, {"helper_sha256": "0" * 64},
                       {"batch_size": 5}, {"runtime": {"pillow": "different"}}):
            with self.subTest(change=change):
                self.edit_metadata("moco", **change)
                with self.assertRaisesRegex(ValueError, "Vision extraction recipe mismatch"):
                    panel.assemble(self.args)
                save(self.args.moco_metadata, original)

    def test_opaque_id_names_do_not_change_any_cost_or_selected_permutation(self):
        panel.assemble(self.args)
        blind = read(self.args.out / "blind-panel.json")
        renamed = copy.deepcopy(blind)
        renamed["image_ids"] = [f"opaque-photo-{i}" for i in range(5)]
        renamed["text_ids"] = [f"opaque-text-{i}" for i in range(5)]
        original = panel.panel_records(blind, core, self.provenance)
        changed = panel.panel_records(renamed, core, self.provenance)
        for pair in PAIRS:
            for old, new in zip(original[pair], changed[pair]):
                for key in ("text_indices", "cost", "distance_correlation", "optimal"):
                    self.assertEqual(old[key], new[key])
                self.assertEqual(new["mapping"], {renamed["image_ids"][i]: renamed["text_ids"][j]
                                                 for i, j in enumerate(new["text_indices"])})

    def test_missing_extra_reordered_axes_and_answer_field_rejected(self):
        panel.assemble(self.args)
        blind = read(self.args.out / "blind-panel.json")
        cases = []
        missing = copy.deepcopy(blind)
        del missing["image_distances"]["mae"]
        cases.append((missing, "Incomplete model panel"))
        extra = copy.deepcopy(blind)
        extra["text_distances"]["unplanned"] = extra["text_distances"]["glove"]
        cases.append((extra, "Incomplete model panel"))
        reordered = copy.deepcopy(blind)
        reordered["image_distances"] = dict(reversed(list(reordered["image_distances"].items())))
        cases.append((reordered, "Incomplete model panel"))
        leaked = copy.deepcopy(blind)
        leaked["mapping"] = {"I00": "T00"}
        cases.append((leaked, "Unexpected blind fields"))
        for value, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    panel.panel_records(value, core, self.provenance)

    def test_malformed_distances_rejected_by_core(self):
        panel.assemble(self.args)
        blind = read(self.args.out / "blind-panel.json")
        for replacement in (float("nan"), 3.0, 0.125):
            value = copy.deepcopy(blind)
            value["image_distances"]["dino"][0][0] = replacement
            with self.subTest(replacement=replacement):
                with self.assertRaises(ValueError):
                    panel.panel_records(value, core, self.provenance)

    def test_changed_selection_rejected_before_ground_truth_is_opened(self):
        self.assemble_and_select()
        (self.selection / "selected.json").write_text("changed")
        self.truth_path.unlink()
        with self.assertRaisesRegex(ValueError, "Selected output changed"):
            self.evaluate()
        self.assertFalse(self.evaluation.exists())

    def test_forged_assignment_with_fresh_checksums_fails_numerical_replay(self):
        self.assemble_and_select()
        path = self.selection / "assignments.json"
        value = read(path)
        value["assignments"]["mae__elmo"][0]["cost"] += 0.001
        save(path, value)
        self.refresh_selection_receipt()
        with self.assertRaisesRegex(ValueError, "Assignment replay mismatch"):
            self.evaluate()

    def test_selected_blind_core_and_pair_bindings_rejected(self):
        self.assemble_and_select()
        path = self.selection / "selected.json"
        original = read(path)
        for change, message in [({"blind_input_sha256": "0" * 64}, "Blind input link changed"),
                                ({"numerical_core_sha256": "0" * 64}, "Numerical core changed"),
                                ({"pairs": {}}, "Incomplete selected panel")]:
            with self.subTest(change=change):
                save(path, {**original, **change})
                self.refresh_selection_receipt()
                with self.assertRaisesRegex(ValueError, message):
                    self.evaluate()
        save(path, original)

    def test_altered_selected_optimum_rejected_even_with_fresh_checksums(self):
        self.assemble_and_select()
        path = self.selection / "selected.json"
        value = read(path)
        value["pairs"]["moco__elmo"]["minimum_cost"] += 0.001
        save(path, value)
        self.refresh_selection_receipt()
        with self.assertRaisesRegex(ValueError, "Selected result mismatch"):
            self.evaluate()

    def test_truth_label_and_mapping_changes_rejected_against_original(self):
        self.assemble_and_select()
        altered = copy.deepcopy(self.truth)
        altered["labels"]["T00"] = "A different synthetic label"
        swapped = copy.deepcopy(self.truth)
        swapped["mapping"]["I00"], swapped["mapping"]["I01"] = swapped["mapping"]["I01"], swapped["mapping"]["I00"]
        for value in (altered, swapped):
            with self.subTest(value=value):
                save(self.truth_path, value)
                with self.assertRaisesRegex(ValueError, "Human label/mapping record changed"):
                    self.evaluate()
                self.assertFalse(self.evaluation.exists())

    def test_primary_evaluation_modified_bytes_rejected(self):
        self.assemble_and_select()
        with self.primary.open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "Original evaluation bytes changed"):
            self.evaluate()

    def test_primary_result_must_replay_even_when_fixture_pin_is_updated(self):
        self.assemble_and_select()
        value = read(self.primary)
        value["true_cost"] += 0.001
        save(self.primary, value)
        with mock.patch.object(panel, "PRIMARY_EVALUATION_SHA", sha(self.primary)):
            with self.assertRaisesRegex(ValueError, "Original DINO/GloVe result did not reproduce exactly"):
                self.evaluate()

    def test_duplicate_keys_nonfinite_json_and_existing_outputs_rejected(self):
        for document in ('{"a": 1, "a": 2}', '{"a": NaN}'):
            path = self.root / "bad.json"
            path.write_text(document)
            with self.assertRaises(ValueError):
                panel.read(path)
        self.assemble_and_select()
        saved = (self.args.out / "features.npz").read_bytes()
        with self.assertRaises(FileExistsError):
            panel.assemble(self.args)
        self.assertEqual((self.args.out / "features.npz").read_bytes(), saved)
        self.evaluate()
        saved = self.evaluation.read_bytes()
        with self.assertRaises(FileExistsError):
            self.evaluate()
        self.assertEqual(self.evaluation.read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
