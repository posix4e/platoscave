#!/usr/bin/env python3
"""Check frozen selection invariants directly against source annotation rows."""
import argparse
import collections
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import random


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).parent
    doc = json.loads(args.manifest.read_text())
    rows = doc["rows"]
    meta = doc["metadata"]
    raw = list(csv.DictReader(args.annotations.open(newline="", encoding="utf-8-sig")))
    source = {r["uid"]: r for r in raw}
    assert sha(args.annotations) == meta["source"]["sha256"]
    assert sha(root / "protocol.json") == meta["protocol_sha256"]
    assert sha(root / "select_rows.py") == meta["selection_script_sha256"]
    assert len(rows) == len({r["uid"] for r in rows}) == 9392
    assert [int(r["uid"]) for r in rows] == sorted(int(r["uid"]) for r in rows)
    assert collections.Counter(r["split"] for r in rows) == {"fit": 6639, "dev": 1127, "test": 1626}
    fit = [r for r in rows if r["split"] == "fit"]
    assert collections.Counter(r["fit_role"] for r in fit) == {"x": 3187, "y": 3452}
    participants = {s: {r["participant_id"] for r in rows if r["split"] == s} for s in ("fit", "dev", "test")}
    assert [len(participants[s]) for s in ("fit", "dev", "test")] == [16, 5, 7]
    assert not (participants["fit"] & participants["dev"] or participants["fit"] & participants["test"] or participants["dev"] & participants["test"])
    xvideos = {r["video_id"] for r in fit if r["fit_role"] == "x"}
    yvideos = {r["video_id"] for r in fit if r["fit_role"] == "y"}
    assert not xvideos & yvideos
    paths = []
    for row in rows:
        original = source[row["uid"]]
        for key in ("participant_id", "video_id", "narration", "noun_class", "verb_class", "start_timestamp", "stop_timestamp"):
            assert row[key] == original[key]
        assert row["start_frame"] == int(original["start_frame"])
        assert row["stop_frame"] == int(original["stop_frame"])
        assert len(row["frame_paths"]) == len(row["frame_times_seconds"]) == 4
        assert all(row["start_seconds"] < t < row["stop_seconds"] for t in row["frame_times_seconds"])
        assert row["frame_times_seconds"] == sorted(set(row["frame_times_seconds"]))
        paths.extend(row["frame_paths"])
    assert len(paths) == len(set(paths)) == 37568
    for person in set(r["participant_id"] for r in rows):
        chosen = [r for r in rows if r["participant_id"] == person]
        original = [r for r in raw if r["participant_id"] == person]
        cap = 512 if chosen[0]["split"] == "fit" else 256
        assert len(chosen) == min(cap, len(original))
        ranks = sorted((hashlib.sha256(f"epic55-clean-v1|row|{r['uid']}".encode()).hexdigest(), int(r["uid"]), r["uid"]) for r in original)
        assert {r["uid"] for r in chosen} == {v[2] for v in ranks[:cap]}
    spec = importlib.util.spec_from_file_location("selector_under_test", root / "select_rows.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    shuffled = list(raw)
    random.Random(20261010).shuffle(shuffled)
    rerun, _ = module.select(shuffled)
    assert rerun == rows, "Input file order must not influence the sample or roles"
    relabeled = [{**r, "noun_class": "999", "verb_class": "999", "narration": "irrelevant placeholder"} for r in shuffled]
    changed, _ = module.select(relabeled)
    identity = lambda data: [(r["uid"], r["split"], r["fit_role"], r["frame_times_seconds"]) for r in data]
    assert identity(changed) == identity(rows), "Semantic contents must not influence selection"
    report = {
        "passed": True,
        "scope": "Metadata selection only; no acquired frame, encoder, alignment or score is verified here",
        "manifest_sha256": sha(args.manifest), "protocol_sha256": sha(root / "protocol.json"),
        "selector_sha256": sha(root / "select_rows.py"), "verification_script_sha256": sha(Path(__file__)),
        "source_sha256": sha(args.annotations), "rows": len(rows), "frame_requests": len(paths),
        "verified": ["Pinned source/protocol/code hashes", "Exact frozen sample counts and caps", "Unique source UIDs and output frame paths", "Participant-disjoint partitions", "Video-disjoint primary fitting marginals", "Retained metadata matches source annotations", "Requested frames lie inside their action windows", "Exact class-blind hash membership", "Invariance to source row order", "Invariance of selection to narration and class-label changes"]
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
