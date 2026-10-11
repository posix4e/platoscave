#!/usr/bin/env python3
"""Acquire the original public COCO images; do not compute model features.

Use a new data directory with verified annotations/captions_{split}.json files.
Each invocation acquires one split. Existing outputs and partial attempts are
preserved. Only the newly downloaded ZIP is removed, after every extracted image
passes ZIP CRC, annotation filename/size checks, and a recorded SHA-256 hash.
The observed hashes record our download; they are not publisher-supplied hashes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import urllib.request
import zipfile


COUNTS = {'train2014': 82783, 'val2014': 40504}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def expected_images(annotation_path, split):
    data = json.loads(annotation_path.read_text())
    images = sorted(data['images'], key=lambda image: image['id'])
    require(len(images) == COUNTS[split], 'Annotation image count differs from published corpus')
    require(len({image['id'] for image in images}) == len(images), 'Duplicate annotation image IDs')
    names = set()
    for image in images:
        name = image['file_name']
        require(isinstance(name, str) and Path(name).name == name and name.endswith('.jpg'),
                'Unsafe or unexpected annotation image name')
        require(name == f'COCO_{split}_{image["id"]:012d}.jpg', 'Image ID and filename disagree')
        names.add(f'{split}/{name}')
    require(len(names) == len(images), 'Duplicate annotation filenames')
    return images, names


def check_archive(archive, expected, split):
    records = archive.infolist()
    require(len({record.filename for record in records}) == len(records), 'Duplicate ZIP member')
    files = {}
    for record in records:
        name = record.filename
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts, 'Unsafe ZIP path')
        if record.is_dir():
            require(name.rstrip('/') == split, 'Unexpected directory in ZIP')
            continue
        require(name in expected and not record.flag_bits & 1, 'Unexpected or encrypted ZIP member')
        mode = record.external_attr >> 16
        require(not stat.S_ISLNK(mode), 'ZIP symlink is not an image')
        require(record.file_size > 0, 'Empty JPEG member')
        files[name] = record
    require(set(files) == expected, 'ZIP image set differs from complete annotations')
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--split', choices=COUNTS, required=True)
    args = parser.parse_args()
    root = args.data.resolve()
    split = args.split
    root.mkdir(parents=True, exist_ok=True)
    annotation_path = root / 'annotations' / f'captions_{split}.json'
    images, expected = expected_images(annotation_path, split)
    annotation_hash = sha256(annotation_path)
    url = f'https://s3.amazonaws.com/images.cocodataset.org/zips/{split}.zip'
    request = urllib.request.Request(url, headers={'User-Agent': 'PlatosCave-research-acquisition'})
    archive_path = root / f'{split}.zip'
    partial = root / f'{split}.zip.partial'
    attempt_path = root / f'{split}-acquisition-start.json'
    for path in (archive_path, partial, root / split, attempt_path, root / f'{split}-images.json',
                 root / f'{split}-acquisition.json'):
        require(not path.exists(), f'Preserving an existing acquisition path: {path.name}')
    with urllib.request.urlopen(request, timeout=90) as response:
        require(response.status == 200 and response.geturl() == url, 'Unexpected image download response')
        length = int(response.headers['Content-Length'])
        require(length > 1_000_000_000, 'Image archive is unexpectedly small')
        require(shutil.disk_usage(root).free >= 2 * length + 2_000_000_000,
                'Insufficient free disk for archive, extraction and safety margin')
        receipt = {'schema_version': 1, 'status': 'acquiring', 'split': split, 'started_at_utc': now(),
                   'upstream_url': f'http://images.cocodataset.org/zips/{split}.zip',
                   'retrieval_url': url, 'final_url': response.geturl(), 'content_length': length,
                   'etag': response.headers.get('ETag'), 'last_modified': response.headers.get('Last-Modified'),
                   'annotation_sha256': annotation_hash, 'expected_images': len(images),
                   'hash_basis': 'Observed download and member hashes; no publisher SHA-256 claim.',
                   'script_sha256': sha256(Path(__file__)), 'scope': 'Public data acquisition only; no feature extraction or fitting.'}
        write_new(attempt_path, receipt)
        digest = hashlib.sha256()
        received = 0
        with partial.open('xb') as stream:
            while block := response.read(8 * 1024 * 1024):
                stream.write(block)
                digest.update(block)
                received += len(block)
            stream.flush()
            os.fsync(stream.fileno())
        require(received == length, 'Incomplete image archive download')
        receipt.update(archive_sha256=digest.hexdigest(), archive_bytes=received, download_finished_at_utc=now())
    partial.rename(archive_path)
    print(json.dumps({'split': split, 'stage': 'downloaded', 'bytes': received}), flush=True)
    image_records = []
    with zipfile.ZipFile(archive_path) as archive:
        members = check_archive(archive, expected, split)
        needed = sum(member.file_size for member in members.values())
        require(shutil.disk_usage(root).free >= needed + 1_000_000_000, 'Insufficient space for extracted images')
        (root / split).mkdir()
        for index, image in enumerate(images):
            name = f'{split}/{image["file_name"]}'
            member = members[name]
            path = root / name
            digest = hashlib.sha256()
            size = 0
            with archive.open(member) as source, path.open('xb') as target:
                while block := source.read(1 << 20):
                    target.write(block)
                    digest.update(block)
                    size += len(block)
            # Reading the complete ZipExtFile above also verifies its stored CRC.
            require(size == member.file_size and path.stat().st_size == size, 'Extracted image size mismatch')
            image_records.append({'row': index, 'image_id': image['id'], 'file': name, 'bytes': size,
                                  'sha256': digest.hexdigest(), 'zip_crc32': f'{member.CRC:08x}'})
            if (index + 1) % 10000 == 0:
                print(json.dumps({'split': split, 'stage': 'extracted', 'images': index + 1}), flush=True)
    require(sha256(annotation_path) == annotation_hash, 'Annotations changed during acquisition')
    # A separate read verifies the recorded hashes of bytes now on disk.
    for record in image_records:
        require(sha256(root / record['file']) == record['sha256'], 'Post-extraction image hash mismatch')
    manifest_path = root / f'{split}-images.json'
    write_new(manifest_path, {'schema_version': 1, 'split': split, 'row_order': 'ascending image ID, as authors loader',
                             'annotation_sha256': annotation_hash, 'images': image_records})
    require(sha256(archive_path) == receipt['archive_sha256'], 'Archive changed during extraction')
    # Only this invocation's verified, now redundant ZIP is removed.
    archive_path.unlink()
    receipt.update(status='complete', completed_at_utc=now(), images=len(image_records),
                   uncompressed_bytes=sum(record['bytes'] for record in image_records),
                   image_manifest_sha256=sha256(manifest_path), all_crc_and_disk_hashes_verified=True,
                   generated_archive_removed_after_verification=True)
    write_new(root / f'{split}-acquisition.json', receipt)
    print(json.dumps({'split': split, 'status': 'complete', 'images': len(image_records)}), flush=True)


if __name__ == '__main__':
    main()
