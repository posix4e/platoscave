#!/usr/bin/env python3
"""Frozen original ELMo top-layer features; the readiness mode uses artificial text only."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import tempfile
import unicodedata


WEIGHTS_SHA256 = "a6b92b9ccbb13fe489faacd2221c795d4bebe84b75f8aded9d85ea464fa5c994"
OPTIONS_SHA256 = "86cb4c1a7e2f25e3867d06810e33ae27fbbd511ac77ca377be5fb7405a7085b1"
WEIGHTS_BYTES = 374434792
OPTIONS_BYTES = 336
BACKEND_SHA256 = {
    "modules/elmo.py": "a02b3ce51773fa125ae889affc21a65201bd8451bee997b3e956d9cb9f0d5c90",
    "modules/elmo_lstm.py": "aec79417f2d60d722847bb861475edafee07bc163a1aeee2f70764d5d5290056",
    "modules/encoder_base.py": "373b14173415ba1180c9f1544ea7762e3b263c698b30bd486d8cb175c529b9d0",
    "modules/lstm_cell_with_projection.py": "66f84517e5fe4ba3b78a955580bc4e206fdb470b2c6d0a05039af5d9c89f03cf",
    "modules/highway.py": "47d270ab4f478426121c476922b1948f995228f171955b270031a5fd3c90daf7",
    "modules/scalar_mix.py": "9efb83375d4ef2d9f56ba464e8db2a13d3e529fd33115096bdf14fd345dc96eb",
    "nn/util.py": "ace9269f2b80170358abb9006ef4c9b3bc7886828c49c24b9e33b1e41397345e",
    "data/token_indexers/elmo_indexer.py": "28e3ef2ff9a9e564c07e574ef63a28f25fe2d9757518971c604ae2b8d7440ae7",
}
EXPECTED_RUNTIME = {"allennlp": "2.10.1", "torch": "1.12.1+cpu", "numpy": "1.23.5", "h5py": "3.8.0"}
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenize(text):
    require(isinstance(text, str), "Every caption must be a string")
    tokens = TOKEN_PATTERN.findall(unicodedata.normalize("NFKC", text).lower())
    require(bool(tokens), "Caption has no tokens after normalization")
    require(all(len(token.encode("utf-8")) <= 48 for token in tokens),
            "A token exceeds ELMo's 48-byte payload; silent character truncation is forbidden")
    return tokens


def load_captions(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(data, dict) and set(data) == {"sample_ids", "captions"},
            "Caption input must contain exactly sample_ids and captions")
    ids, captions = data["sample_ids"], data["captions"]
    require(isinstance(ids, list) and isinstance(captions, list) and 0 < len(ids) == len(captions),
            "Caption IDs and text must be equal nonempty lists")
    require(all(isinstance(value, str) and value.strip() for value in ids) and len(ids) == len(set(ids)),
            "Caption IDs must be nonempty unique strings")
    return ids, [tokenize(caption) for caption in captions]


class FrozenElmo:
    def __init__(self, options, weights, threads=4):
        self.options, self.weights = Path(options), Path(weights)
        require(self.options.stat().st_size == OPTIONS_BYTES and sha256(self.options) == OPTIONS_SHA256,
                "Original ELMo options hash/size mismatch")
        require(self.weights.stat().st_size == WEIGHTS_BYTES and sha256(self.weights) == WEIGHTS_SHA256,
                "Original ELMo weights hash/size mismatch")
        self.runtime = {name: importlib.metadata.version(name) for name in EXPECTED_RUNTIME}
        require(self.runtime == EXPECTED_RUNTIME, "Runtime differs from the pinned CPU ELMo stack")
        distribution = importlib.metadata.distribution("allennlp")
        self.backend = Path(distribution.locate_file("allennlp"))
        self.check_sources()
        import numpy as np
        import torch
        from allennlp.modules.elmo import _ElmoBiLm, batch_to_ids
        from allennlp.nn.util import remove_sentence_boundaries

        self.np, self.torch = np, torch
        self.batch_to_ids, self.remove_boundaries = batch_to_ids, remove_sentence_boundaries
        require(threads > 0, "Thread count must be positive")
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        self.model = _ElmoBiLm(str(self.options), str(self.weights), requires_grad=False)
        self.model.requires_grad_(False)
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
            if hasattr(module, "recurrent_dropout_probability"):
                require(module.recurrent_dropout_probability == 0.0, "Unexpected recurrent dropout")
        require(not any(parameter.requires_grad for parameter in self.model.parameters()), "Model is not frozen")
        require(self.model.num_layers == 3 and self.model.get_output_dim() == 1024, "Unexpected original ELMo shape")
        self.runtime.update({"python": platform.python_version(), "system": platform.system(),
                             "machine": platform.machine(), "device": "cpu", "threads": threads,
                             "interop_threads": 1, "deterministic_algorithms": True})

    def check_sources(self):
        for name, expected in BACKEND_SHA256.items():
            require(sha256(self.backend / name) == expected, f"Official AllenNLP source mismatch: {name}")

    def batch(self, token_lists):
        """Fresh state for every row in the batch; production calls this with one row."""
        torch, np = self.torch, self.np
        require(bool(token_lists) and all(tokens for tokens in token_lists), "Cannot encode empty tokens")
        require(all(len(token.encode("utf-8")) <= 48 for tokens in token_lists for token in tokens),
                "Character truncation is forbidden")
        self.model._elmo_lstm.reset_states()
        require(self.model._elmo_lstm._states is None, "ELMo recurrent state did not reset")
        with torch.inference_mode():
            encoded = self.model(self.batch_to_ids(token_lists))
            top, mask = self.remove_boundaries(encoded["activations"][-1], encoded["mask"])
            lengths = mask.sum(dim=1)
            require(lengths.tolist() == [len(tokens) for tokens in token_lists],
                    "Boundary removal or padding mask lost real tokens")
            pooled = (top * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(-1)
            norm = torch.linalg.vector_norm(pooled, dim=1, keepdim=True)
            require(bool(torch.isfinite(pooled).all()) and bool((norm > 0).all()),
                    "ELMo produced zero or nonfinite vectors")
            features = (pooled / norm).cpu().numpy().astype(np.float32)
        require(features.shape == (len(token_lists), 1024) and bool(np.isfinite(features).all()),
                "Invalid feature array")
        require(bool(np.allclose(np.linalg.norm(features.astype(np.float64), axis=1), 1, rtol=0, atol=1e-5)),
                "ELMo feature normalization failed")
        return features

    def independent(self, token_lists):
        return self.np.concatenate([self.batch([tokens]) for tokens in token_lists], axis=0)

    def metadata(self):
        self.check_sources()
        require(sha256(self.weights) == WEIGHTS_SHA256 and sha256(self.options) == OPTIONS_SHA256,
                "Model input changed during extraction")
        return {
            "schema_version": 1, "status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": sha256(Path(__file__)), "weights_sha256": WEIGHTS_SHA256,
            "options_sha256": OPTIONS_SHA256, "backend_source_sha256": BACKEND_SHA256,
            "backend_reference": "https://github.com/allenai/allennlp/tree/v2.10.1",
            "runtime": self.runtime, "output_dtype": "float32", "feature_dimensions": 1024,
            "frozen_model": True, "all_finite": True, "l2_normalized": True,
            "tokenization": "Unicode NFKC, lowercase, regex [a-z]+(?:'[a-z]+)?|[0-9]+; original token order retained",
            "representation": "Final contextual layer, mean of real tokens excluding BOS/EOS/padding, then L2",
            "recurrent_state": "reset_states before every independent sentence; production batch size one",
            "dropout": 0.0, "scalar_mix": "none; _ElmoBiLm activations[-1] used directly",
            "character_truncation": "Rejected when a token exceeds 48 UTF-8 bytes",
        }


def save_json_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.part-", dir=path.parent)
    staging = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.link(staging, path)
    finally:
        staging.unlink()


def readiness(encoder):
    np = encoder.np
    captions = ["Quiet letters turn slowly.", "Readers compare two passages.",
                "An unfinished sentence.", "Brief."]
    tokens = [tokenize(caption) for caption in captions]
    first = encoder.independent(tokens)
    again = encoder.independent(tokens)
    reversed_back = encoder.independent(tokens[::-1])[::-1]
    batched = encoder.batch(tokens)
    batched_reversed = encoder.batch(tokens[::-1])[::-1]
    padded_companion = encoder.batch([tokens[3], tokens[0]])[0]
    require(np.array_equal(first, again) and np.array_equal(first, reversed_back),
            "Individual inference depends on earlier sentences or sentence order")
    errors = {"all_at_once": float(np.max(np.abs(first - batched))),
              "reverse_batch": float(np.max(np.abs(first - batched_reversed))),
              "padded_companion": float(np.max(np.abs(first[3] - padded_companion)))}
    require(all(value <= 1e-5 for value in errors.values()), "Batch or padding invariance check failed")
    for invalid in ("", "☀", "a" * 49):
        try:
            tokenize(invalid)
        except ValueError:
            pass
        else:
            raise ValueError("Empty-input or character-truncation guard failed")
    require(tokenize(" ＡＢＣ don't 123 ") == ["abc", "don't", "123"], "Tokenizer rule differs")
    return {**encoder.metadata(), "status": "synthetic_readiness_passed", "sample_count": len(captions),
            "synthetic_captions": captions, "actual_evaluation_text_used": False,
            "repeat_individual_bit_identical": True, "reordered_individual_bit_identical": True,
            "batch_invariance_max_abs": errors, "batch_invariance_tolerance": 1e-5,
            "empty_and_truncation_guards_passed": True, "shape": list(first.shape),
            "limits": "Implementation readiness only; these artificial sentences are not evaluation data."}


def extract(captions_path, encoder, output):
    output, captions_path = Path(output), Path(captions_path)
    require(output.suffix == ".npz", "Feature output must end in .npz")
    sidecar = output.with_suffix(".json")
    require(not output.exists() and not sidecar.exists(), "Feature or metadata output already exists")
    original_sha = sha256(captions_path)
    ids, tokens = load_captions(captions_path)
    features = encoder.independent(tokens)
    metadata = {**encoder.metadata(), "sample_count": len(ids), "sample_ids": ids,
                "captions_sha256": original_sha,
                "tokens": [{"sample_id": sample_id, "tokens": row, "count": len(row)}
                           for sample_id, row in zip(ids, tokens)]}
    require(sha256(captions_path) == original_sha, "Caption input changed during extraction")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.part-", dir=output.parent)
    staging, published = Path(temporary), False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            encoder.np.savez_compressed(stream, features=features, sample_ids=encoder.np.asarray(ids, dtype=str))
        metadata["feature_sha256"] = sha256(staging)
        os.link(staging, output)
        published = True
        save_json_exclusive(sidecar, metadata)
    except Exception:
        if published:
            output.unlink()
        raise
    finally:
        staging.unlink()
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--options", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    require(args.self_test != bool(args.captions), "Choose either --self-test or --captions")
    require(not args.out.exists(), "Output already exists")
    if args.captions:
        load_captions(args.captions)
        require(not args.out.with_suffix(".json").exists(), "Metadata output already exists")
    encoder = FrozenElmo(args.options, args.weights, args.threads)
    if args.self_test:
        metadata = readiness(encoder)
        save_json_exclusive(args.out, metadata)
    else:
        metadata = extract(args.captions, encoder, args.out)
    print(json.dumps({"status": metadata["status"], "sample_count": metadata["sample_count"],
                      "source_sha256": metadata["source_sha256"],
                      "feature_sha256": metadata.get("feature_sha256")}, indent=2))


if __name__ == "__main__":
    main()
