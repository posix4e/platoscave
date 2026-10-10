#!/usr/bin/env python3
"""Audit released caption groups against COCO raw captions, without dependencies.

This measures membership in a released corpus, never actual checkpoint sampling.
One candidate source is uniqueness within the reference, not proof of provenance.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import zipfile


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def source_identity(image: dict) -> tuple[int, str]:
    """Use original COCO release identity; deliberately ignore Karpathy split."""
    filename = image.get("filename", "")
    match = re.fullmatch(r"COCO_(train2014|val2014)_(\d+)\.jpg", filename)
    if not match:
        raise ValueError(f"Unrecognized COCO filename: {filename!r}")
    split, source_id = match.group(1), int(match.group(2))
    filepath = image.get("filepath")
    if filepath is not None and filepath != split:
        raise ValueError(f"Conflicting filepath/filename for {filename!r}")
    if image.get("cocoid") != source_id:
        raise ValueError(f"Conflicting cocoid/filename for {filename!r}")
    return source_id, split


def load_reference(path: Path, member: str):
    with zipfile.ZipFile(path) as archive:
        raw = archive.read(member)
    data = json.loads(raw)
    indices = {"exact": defaultdict(set), "normalized": defaultdict(set)}
    sources = {}
    caption_counts = Counter()
    karpathy_counts = Counter()
    for image in data["images"]:
        source_id, split = source_identity(image)
        if source_id in sources:
            raise ValueError(f"Duplicate reference source ID: {source_id}")
        sources[source_id] = split
        karpathy_counts[str(image.get("split", "missing"))] += 1
        for sentence in image["sentences"]:
            caption = sentence["raw"]
            if not isinstance(caption, str) or not normalize(caption):
                raise ValueError(f"Invalid raw caption for source {source_id}")
            caption_counts[split] += 1
            indices["exact"][caption].add(source_id)
            indices["normalized"][normalize(caption)].add(source_id)
    metadata = {
        "archive_sha256": sha256_file(path),
        "member": member,
        "member_sha256": hashlib.sha256(raw).hexdigest(),
        "images": len(sources),
        "images_by_original_split": dict(sorted(Counter(sources.values()).items())),
        "caption_occurrences_by_original_split": dict(sorted(caption_counts.items())),
        "karpathy_split_counts_diagnostic_only": dict(sorted(karpathy_counts.items())),
        "unique_captions": {mode: len(index) for mode, index in indices.items()},
    }
    return indices, sources, metadata


class ModeAudit:
    def __init__(self, index, sources):
        self.index = index
        self.sources = sources
        self.seen = set()
        self.matched = set()
        self.text_occurrences = 0
        self.matched_occurrences = 0
        self.rows_with_any_match = 0
        self.rows_with_all_match = 0
        self.rows_single_distinct_text = 0
        self.candidate_histogram = Counter()
        self.candidate_histogram_distinct = Counter()
        self.unique_rows_by_split = Counter()
        self.unique_distinct_rows_by_split = Counter()
        self.guid_counts = Counter()
        self.guid_known_rows_by_split = Counter()
        self.guid_consistent_rows_by_split = Counter()
        self.image_sets = {
            "any_matching_caption": set(),
            "any_whole_group_candidate": set(),
            "unique_whole_group_candidate": set(),
            "unique_whole_group_candidate_at_least_two_distinct_texts": set(),
            "ambiguous_whole_group_candidate": set(),
            "numeric_guid_known_source": set(),
            "numeric_guid_source_with_all_captions_matching": set(),
        }

    def add(self, texts: list[str], guid_status="absent", guid_id=None) -> dict:
        self.text_occurrences += len(texts)
        self.seen.update(texts)
        distinct = set(texts)
        matches = {text: self.index.get(text, set()) for text in distinct}
        occurrences = sum(bool(matches[text]) for text in texts)
        self.matched_occurrences += occurrences
        self.matched.update(text for text, ids in matches.items() if ids)
        self.rows_with_any_match += bool(occurrences)
        self.rows_with_all_match += occurrences == len(texts)
        self.rows_single_distinct_text += len(distinct) == 1
        any_sources = set().union(*matches.values())
        self.image_sets["any_matching_caption"].update(any_sources)
        # Intersection across EVERY distinct text. An absent text forces zero.
        candidates = set.intersection(*(set(ids) for ids in matches.values()))
        count = len(candidates)
        self.candidate_histogram[count] += 1
        if len(distinct) >= 2:
            self.candidate_histogram_distinct[count] += 1
        self.image_sets["any_whole_group_candidate"].update(candidates)
        if count == 1:
            source_id = next(iter(candidates))
            self.image_sets["unique_whole_group_candidate"].add(source_id)
            self.unique_rows_by_split[self.sources[source_id]] += 1
            if len(distinct) >= 2:
                self.image_sets["unique_whole_group_candidate_at_least_two_distinct_texts"].add(source_id)
                self.unique_distinct_rows_by_split[self.sources[source_id]] += 1
        elif count > 1:
            self.image_sets["ambiguous_whole_group_candidate"].update(candidates)
        guid_detail = {"status": guid_status}
        if guid_status == "known_reference_source":
            split = self.sources[guid_id]
            consistent = guid_id in candidates
            guid_status = "known_source_all_captions_match" if consistent else "known_source_caption_mismatch"
            self.guid_known_rows_by_split[split] += 1
            self.image_sets["numeric_guid_known_source"].add(guid_id)
            if consistent:
                self.guid_consistent_rows_by_split[split] += 1
                self.image_sets["numeric_guid_source_with_all_captions_matching"].add(guid_id)
            guid_detail = {"status": guid_status, "source_id": guid_id,
                           "original_split": split, "all_captions_match_this_source": consistent}
        self.guid_counts[guid_status] += 1
        return {
            "distinct_texts": len(distinct),
            "matching_text_occurrences": occurrences,
            "candidate_source_ids": sorted(candidates),
            "candidate_source_count": count,
            "all_texts_match_somewhere": occurrences == len(texts),
            "candidate_class": "none" if not count else "unique" if count == 1 else "ambiguous",
            "guid_cross_check": guid_detail,
        }

    def report(self) -> dict:
        images = {}
        for label, ids in self.image_sets.items():
            by_split = {split: sorted(i for i in ids if self.sources[i] == split)
                        for split in ("train2014", "val2014")}
            images[label] = {
                "count": len(ids),
                "counts_by_original_split": {split: len(values) for split, values in by_split.items()},
                "source_ids_by_original_split": by_split,
            }
        return {
            "unique_corpus_captions": len(self.seen),
            "unique_matching_corpus_captions": len(self.matched),
            "text_occurrences": self.text_occurrences,
            "matching_text_occurrences": self.matched_occurrences,
            "rows_with_any_text_match": self.rows_with_any_match,
            "rows_with_all_texts_matching_somewhere": self.rows_with_all_match,
            "rows_with_only_one_distinct_text": self.rows_single_distinct_text,
            "whole_group_candidate_count_histogram": dict(sorted(self.candidate_histogram.items())),
            "whole_group_candidate_count_histogram_at_least_two_distinct_texts": dict(sorted(self.candidate_histogram_distinct.items())),
            "unique_candidate_rows_by_original_split": dict(sorted(self.unique_rows_by_split.items())),
            "unique_candidate_rows_at_least_two_distinct_texts_by_original_split": dict(sorted(self.unique_distinct_rows_by_split.items())),
            "guid_diagnostic": {
                "row_counts_by_status": dict(sorted(self.guid_counts.items())),
                "known_source_rows_by_original_split": dict(sorted(self.guid_known_rows_by_split.items())),
                "known_source_all_captions_match_rows_by_original_split": dict(sorted(self.guid_consistent_rows_by_split.items())),
            },
            "matching_source_images": images,
        }


def parse_group(value, line_number: int) -> list[str]:
    if isinstance(value, dict):
        value = value.get("texts")
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(f"Corpus line {line_number}: expected >=2 strings or {{'texts': [...]}}")
    if any(not isinstance(text, str) or not normalize(text) for text in value):
        raise ValueError(f"Corpus line {line_number}: invalid or empty caption")
    return value


def classify_guid(value, sources):
    if not isinstance(value, dict) or "guid" not in value:
        return "absent", None
    guid = value["guid"]
    if isinstance(guid, str) and re.fullmatch(r"[0-9]+", guid):
        source_id = int(guid)
    elif isinstance(guid, int) and not isinstance(guid, bool) and guid >= 0:
        source_id = guid
    else:
        return "nonnumeric", None
    if source_id not in sources:
        return "numeric_source_absent_from_reference", source_id
    return "known_reference_source", source_id


def audit(corpus: Path, reference: Path, member: str = "dataset_coco.json",
          row_evidence: Path | None = None) -> dict:
    indices, sources, reference_metadata = load_reference(reference, member)
    modes = {mode: ModeAudit(index, sources) for mode, index in indices.items()}
    digest = hashlib.sha256()
    rows = 0
    sizes = Counter()
    evidence = None
    if row_evidence is not None:
        row_evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence = row_evidence.open("w", encoding="utf-8", newline="\n")
    try:
        with gzip.open(corpus, "rb") as stream:
            for number, raw_line in enumerate(stream, 1):
                digest.update(raw_line)
                if not raw_line.strip():
                    raise ValueError(f"Corpus line {number}: blank JSONL line")
                value = json.loads(raw_line.decode("utf-8"))
                texts = parse_group(value, number)
                guid_status, guid_id = classify_guid(value, sources)
                rows += 1
                sizes[len(texts)] += 1
                detail = {
                    "row_1based": number,
                    "exact": modes["exact"].add(texts, guid_status, guid_id),
                    "normalized": modes["normalized"].add([normalize(text) for text in texts], guid_status, guid_id),
                }
                if isinstance(value, dict) and "guid" in value:
                    # Retain release metadata alongside, not replaced by, the
                    # diagnostic numeric-ID/caption-consistency cross-check.
                    detail["corpus_guid_opaque"] = value["guid"]
                if evidence:
                    evidence.write(json.dumps(detail, sort_keys=True) + "\n")
    finally:
        if evidence:
            evidence.close()
    result = {
        "schema_version": "1.0.0",
        "audit_kind": "released_caption_corpus_membership_not_checkpoint_sampling",
        "script_sha256": sha256_file(Path(__file__)),
        "normalization": "Unicode NFKC, then casefold, then collapse/trim Python Unicode whitespace; no punctuation removal",
        "unicode_version": unicodedata.unidata_version,
        "corpus": {
            "filename": corpus.name,
            "compressed_sha256": sha256_file(corpus),
            "uncompressed_sha256": digest.hexdigest(),
            "rows": rows,
            "texts_per_row_histogram": dict(sorted(sizes.items())),
        },
        "reference": reference_metadata,
        "modes": {mode: state.report() for mode, state in modes.items()},
        "interpretation": [
            "Matches establish caption-text membership in the released corpus, not whether a checkpoint actually sampled these rows.",
            "A whole-group candidate is a Karpathy source image containing every distinct group caption; no matching text is silently dropped from this intersection.",
            "Unique means one candidate inside this reference, not proof of original source identity or absence of duplicates outside the reference.",
            "Identical generic captions may occur on multiple images; individual-text matches and ambiguous groups are not definite image exposure.",
            "Groups collapsing to one distinct text are reported separately even when their reference candidate is unique.",
            "The guid diagnostic tests whether a numeric release guid equals a reference COCO ID and every group caption occurs on that image. It reports consistency, not independent proof of guid semantics or checkpoint sampling.",
            "Original COCO train2014/val2014 comes from agreeing filename/filepath metadata; Karpathy train/restval/val/test is not substituted.",
            "This is a text-identity audit, not an image-byte, perceptual-duplicate, training-log, or full pretraining-lineage audit.",
        ],
    }
    if row_evidence is not None:
        result["row_evidence"] = {"filename": row_evidence.name, "sha256": sha256_file(row_evidence)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path, help="Released gzip JSONL caption groups")
    parser.add_argument("--karpathy-zip", required=True, type=Path)
    parser.add_argument("--member", default="dataset_coco.json")
    parser.add_argument("--output", required=True, type=Path, help="JSON report including source-ID lists")
    parser.add_argument("--row-evidence", type=Path, help="Optional JSONL candidate sets for every input row")
    args = parser.parse_args()
    inputs = {args.corpus.resolve(), args.karpathy_zip.resolve(), Path(__file__).resolve()}
    outputs = [args.output.resolve()] + ([args.row_evidence.resolve()] if args.row_evidence else [])
    if len(set(outputs)) != len(outputs) or any(path in inputs for path in outputs):
        parser.error("Outputs must be distinct and must not overwrite inputs or this script")
    result = audit(args.corpus, args.karpathy_zip, args.member, args.row_evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": result["corpus"]["rows"]}))


if __name__ == "__main__":
    main()
