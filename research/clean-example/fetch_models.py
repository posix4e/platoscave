#!/usr/bin/env python3
"""Fetch original DINO/GloVe artifacts and record bytes, without loading models.

Run in the experiment VM. An existing file is rehashed, never silently trusted.
Pass the published artifact-lock.json to verify a repeat acquisition.
"""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile


SOURCES = {
    "dino_vitbase16_pretrain.pth": "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
    "dino_vitbase16_args.json": "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/args.txt",
    "glove.6B.zip": "https://nlp.stanford.edu/data/wordvecs/glove.6B.zip",
}
CHUNK = 1024 * 1024


def digest_stream(stream):
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(CHUNK):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--expected-manifest", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    expected = json.loads(args.expected_manifest.read_text()) if args.expected_manifest else None
    records = {}
    for name, url in SOURCES.items():
        target = args.directory / name
        if not target.exists():
            print(f"Downloading {name}", flush=True)
            temporary = target.with_suffix(target.suffix + ".partial")
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as out:
                count = 0
                announced = 0
                while chunk := response.read(CHUNK):
                    out.write(chunk)
                    count += len(chunk)
                    if count - announced >= 100 * CHUNK:
                        print(f"{name}: {count // CHUNK} MiB", flush=True)
                        announced = count
                length = response.headers.get("Content-Length")
                if length is not None and count != int(length):
                    raise RuntimeError(f"Incomplete download: {name}")
            temporary.replace(target)
        with target.open("rb") as stream:
            digest, size = digest_stream(stream)
        records[name] = {"url": url, "sha256": digest, "bytes": size}
        if expected and records[name] != expected["artifacts"][name]:
            raise RuntimeError(f"Artifact differs from expected manifest: {name}")
        print(f"Hashed {name}: {digest}", flush=True)
    with zipfile.ZipFile(args.directory / "glove.6B.zip") as archive:
        member_name = "glove.6B.300d.txt"
        with archive.open(member_name) as stream:
            digest, size = digest_stream(stream)
        member = {"name": member_name, "sha256": digest, "bytes": size}
        if expected and member != expected["glove_300d_member"]:
            raise RuntimeError("GloVe member differs from expected manifest")
    report = {
        "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "artifacts": records,
        "glove_300d_member": member,
        "executed_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "verified_against_manifest": expected is not None,
        "scope": "Artifact-byte identity only; no inference, training, or independent authentication of training provenance.",
    }
    (args.directory / "artifact-lock.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Saved artifact-lock.json", flush=True)


if __name__ == "__main__":
    main()
