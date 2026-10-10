#!/usr/bin/env python3
"""Freeze a class-blind EPIC-55 sample and participant/video-disjoint roles.

Uses only pinned annotation metadata. No images, vectors, model scores, noun
labels, or verb labels influence selection or splitting.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import unicodedata


SALT = "epic55-clean-v1"
SOURCE_SHA256 = "faef8ad29b163b28ec8ca32b4222f137aba8f8b07eca0ae004652937ffaf811e"
SOURCE_COMMIT = "63fa7d78eba41a8d2280374061a6b1bab3e7e41d"
EXPECTED_PEOPLE = {
    "fit": ["P20", "P12", "P28", "P05", "P29", "P19", "P01", "P14", "P23", "P27", "P26", "P04", "P15", "P10", "P21", "P31"],
    "dev": ["P03", "P06", "P30", "P16", "P24"],
    "test": ["P13", "P22", "P25", "P02", "P08", "P17", "P07"],
}


def digest(domain: str, value: str) -> str:
    return hashlib.sha256(f"{SALT}|{domain}|{value}".encode()).hexdigest()


def seconds(timestamp: str) -> float:
    hours, minutes, remainder = timestamp.split(":")
    return 3600 * int(hours) + 60 * int(minutes) + float(remainder)


def normalized_caption(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def select(rows: list[dict]) -> tuple[list[dict], dict]:
    if len({r["uid"] for r in rows}) != len(rows):
        raise ValueError("Duplicate source UID")
    by_participant = defaultdict(list)
    for row in rows:
        by_participant[row["participant_id"]].append(row)
    people = sorted(by_participant, key=lambda p: digest("participant", p))
    if len(people) != 28:
        raise ValueError("Expected the 28 public-training participants")
    split_people = {"fit": people[:16], "dev": people[16:21], "test": people[21:]}
    if split_people != EXPECTED_PEOPLE:
        raise ValueError("Participant split differs from the frozen protocol")
    selected = []
    participant_counts = {}
    for split, participants in split_people.items():
        cap = 512 if split == "fit" else 256
        for participant in participants:
            subset = sorted(by_participant[participant], key=lambda r: (digest("row", r["uid"]), int(r["uid"])))[:cap]
            selected_by_video = defaultdict(list)
            for row in subset:
                selected_by_video[row["video_id"]].append(row)
            roles = {}
            role_counts = Counter({"x": 0, "y": 0})
            if split == "fit":
                if len(selected_by_video) < 2:
                    raise ValueError(f"{participant} has fewer than two selected videos")
                videos = sorted(selected_by_video, key=lambda v: (-len(selected_by_video[v]), digest("role-video", f"{participant}|{v}")))
                for video in videos:
                    if role_counts["x"] == role_counts["y"]:
                        role = "x" if int(digest("role-tie", f"{participant}|{video}"), 16) % 2 == 0 else "y"
                    else:
                        role = min(("x", "y"), key=lambda k: role_counts[k])
                    roles[video] = role
                    role_counts[role] += len(selected_by_video[video])
                if not all(role_counts[k] for k in ("x", "y")):
                    raise ValueError(f"{participant} lacks one fitting marginal")
            participant_counts[participant] = {
                "split": split, "source_rows": len(by_participant[participant]),
                "selected_rows": len(subset), "selected_videos": len(selected_by_video),
                "fit_role_counts": dict(role_counts) if split == "fit" else {},
            }
            for row in subset:
                start, stop = seconds(row["start_timestamp"]), seconds(row["stop_timestamp"])
                if not (0 <= start < stop):
                    raise ValueError(f"Nonpositive/invalid time interval for UID {row['uid']}")
                times = [round(start + (j + 0.5) * (stop - start) / 4, 8) for j in range(4)]
                selected.append({
                    "uid": row["uid"], "participant_id": participant, "video_id": row["video_id"],
                    "split": split, "fit_role": roles.get(row["video_id"], ""),
                    "narration": row["narration"], "normalized_caption": normalized_caption(row["narration"]),
                    "noun_class": row["noun_class"], "verb_class": row["verb_class"],
                    "start_timestamp": row["start_timestamp"], "stop_timestamp": row["stop_timestamp"],
                    "start_seconds": start, "stop_seconds": stop,
                    "start_frame": int(row["start_frame"]), "stop_frame": int(row["stop_frame"]),
                    "frame_times_seconds": times,
                    "frame_paths": [f"frames/{row['video_id']}/{row['uid']}_f{j}.jpg" for j in range(4)],
                })
    selected.sort(key=lambda r: int(r["uid"]))
    video_roles = defaultdict(set)
    for row in selected:
        if row["split"] == "fit":
            video_roles[row["video_id"]].add(row["fit_role"])
    if any(len(roles) != 1 for roles in video_roles.values()):
        raise AssertionError("A fitting video occurs in both modalities")
    summary = {
        "participants": split_people, "participant_counts": participant_counts,
        "counts_by_split": dict(Counter(r["split"] for r in selected)),
        "fit_counts_by_role": dict(Counter(r["fit_role"] for r in selected if r["split"] == "fit")),
        "fit_video_counts_by_role": dict(Counter(next(iter(roles)) for roles in video_roles.values())),
        "selected_video_count": len({r["video_id"] for r in selected}),
        "selected_row_count": len(selected), "requested_frame_count": 4 * len(selected),
        "fit_video_overlap_between_modalities": 0,
        "participant_overlap_between_splits": 0,
        "row_order": "ascending numeric UID; embedding acquisition must preserve this order",
    }
    return selected, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    args = parser.parse_args()
    raw = args.annotations.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    if source_sha != SOURCE_SHA256:
        raise ValueError(f"Annotation source hash differs: {source_sha}")
    with args.annotations.open(newline="", encoding="utf-8-sig") as stream:
        rows, summary = select(list(csv.DictReader(stream)))
    document = {"metadata": {
        "schema_version": 1, "protocol_id": SALT,
        "source": {"filename": args.annotations.name, "sha256": source_sha,
                   "repository_commit": SOURCE_COMMIT,
                   "url": f"https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-55-annotations/{SOURCE_COMMIT}/EPIC_train_action_labels.csv"},
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "selection_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selection_is_class_blind": True,
        "acquisition_status": "requested; frames have not been checked by this metadata-only selector",
        **summary,
    }, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({**summary, "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
