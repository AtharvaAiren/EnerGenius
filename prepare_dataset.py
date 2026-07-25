from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from contextlib import ExitStack
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path

from room_control.dataset import (
    DATASET_DOI,
    DATASET_TITLE,
    NORMALIZED_FIELDS,
    PAPER_DOI,
    SOURCE_ARCHIVE_MD5,
    SOURCE_ARCHIVE_URL,
)


STREAM_FILES = {
    "co2_ppm": "co2_room_{room}.csv",
    "humidity_pct": "humidity_room_{room}.csv",
    "illuminance_lux": "Illuminance_room_{room}.csv",
    "occupancy": "occupant_count_room_{room}.csv",
    "temperature_c": "temperature_room_{room}.csv",
    "vav_damper_pct": "vav_room_{room}.csv",
}
KEY_FIELDS = ("Year", "Month", "Time", "DayId", "Workday")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and normalize the SDU real-room sensing dataset."
    )
    parser.add_argument(
        "--output",
        default="data/real_room/observations.csv.gz",
        help="Normalized one-minute CSV output.",
    )
    parser.add_argument(
        "--archive",
        help="Use an existing Dataset.zip instead of downloading from Figshare.",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing outputs.")
    return parser.parse_args()


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".part")
    print(f"Downloading {SOURCE_ARCHIVE_URL}")
    with urllib.request.urlopen(SOURCE_ARCHIVE_URL, timeout=60) as response:
        with temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
    temporary.replace(destination)


def safe_extract(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as zipped:
        destination_root = destination.resolve()
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if destination_root not in target.parents and target != destination_root:
                raise ValueError(f"unsafe archive member: {member.filename}")
        zipped.extractall(destination)


def read_room_info(path: Path) -> dict[int, dict[str, object]]:
    rooms: dict[int, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            room_id, room_type, area, capacity, volume = line.split()
            number = int(room_id.split("-")[-1])
            rooms[number] = {
                "room_id": room_id.lower().replace("-", "_"),
                "room_type": room_type,
                "area_m2": float(area),
                "seating_capacity": int(capacity),
                "volume_m3": float(volume),
            }
    if set(rooms) != {1, 2, 3}:
        raise ValueError("expected metadata for rooms 1, 2, and 3")
    return rooms


def stream_reader(handle, room: int):
    reader = csv.DictReader(handle)
    expected_value = f"room_{room}"
    expected_fields = (*KEY_FIELDS, expected_value)
    if tuple(reader.fieldnames or ()) != expected_fields:
        raise ValueError(f"unexpected source schema: {reader.fieldnames}")
    return reader, expected_value


def minute_of_day(value: str) -> int:
    hour, minute, _second = (int(part) for part in value.split(":"))
    return hour * 60 + minute


def normalize(archive: Path, output: Path) -> int:
    with tempfile.TemporaryDirectory(prefix="real-room-dataset-") as directory:
        extracted = Path(directory)
        safe_extract(archive, extracted)
        room_info = read_room_info(extracted / "roominfo.txt")
        source_directory = extracted / "filleddata"
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = output.with_suffix(output.suffix + ".part")
        records = 0
        with gzip.open(temporary_output, "wt", encoding="utf-8", newline="") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=NORMALIZED_FIELDS)
            writer.writeheader()
            for room in (1, 2, 3):
                with ExitStack() as stack:
                    readers: dict[str, tuple[object, str]] = {}
                    for feature, pattern in STREAM_FILES.items():
                        path = source_directory / pattern.format(room=room)
                        handle = stack.enter_context(path.open("r", encoding="utf-8", newline=""))
                        readers[feature] = stream_reader(handle, room)

                    ordered_features = tuple(STREAM_FILES)
                    iterators = [readers[feature][0] for feature in ordered_features]
                    for source_rows in zip_longest(*iterators):
                        if any(row is None for row in source_rows):
                            raise ValueError(f"source streams have unequal lengths for room {room}")
                        first = source_rows[0]
                        key = tuple(first[field] for field in KEY_FIELDS)
                        if any(tuple(row[field] for field in KEY_FIELDS) != key for row in source_rows[1:]):
                            raise ValueError(f"source streams are misaligned for room {room}: {key}")

                        values = {
                            feature: float(row[readers[feature][1]])
                            for feature, row in zip(ordered_features, source_rows)
                        }
                        metadata = room_info[room]
                        writer.writerow(
                            {
                                "room_id": metadata["room_id"],
                                "room_type": metadata["room_type"],
                                "area_m2": metadata["area_m2"],
                                "volume_m3": metadata["volume_m3"],
                                "seating_capacity": metadata["seating_capacity"],
                                "year": int(first["Year"]),
                                "month": int(first["Month"]),
                                "day_id": int(first["DayId"]),
                                "minute_of_day": minute_of_day(first["Time"]),
                                "is_workday": first["Workday"].strip().lower() == "true",
                                "occupancy": values["occupancy"],
                                "temperature_c": values["temperature_c"],
                                "humidity_pct": values["humidity_pct"],
                                "co2_ppm": values["co2_ppm"],
                                "illuminance_lux": values["illuminance_lux"],
                                "vav_damper_pct": values["vav_damper_pct"],
                            }
                        )
                        records += 1
        temporary_output.replace(output)
        return records


def write_metadata(output: Path, archive: Path, records: int) -> Path:
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    payload = {
        "title": DATASET_TITLE,
        "dataset_doi": DATASET_DOI,
        "paper_doi": PAPER_DOI,
        "license": "CC0",
        "source_archive_url": SOURCE_ARCHIVE_URL,
        "source_archive_md5": md5sum(archive),
        "normalized_records": records,
        "rooms": 3,
        "day_ids": 44,
        "sampling_minutes": 1,
        "source_variant": "filleddata (forward/backward-filled by dataset authors)",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "causal_limit": (
            "VAV damper position is the only logged control input. The dataset does not "
            "contain physical fan PWM, LED PWM, device energy, or comfort votes."
        ),
    }
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    if not args.force and output.exists() and metadata_path.exists():
        print(f"Normalized dataset already exists: {output.resolve()}")
        print("Use --force to rebuild it from the source archive.")
        return
    if not args.force and (output.exists() or metadata_path.exists()):
        raise SystemExit(
            "only one normalized output artifact exists; use --force to rebuild both"
        )

    if args.archive:
        archive = Path(args.archive)
    else:
        archive = output.parent / "source" / "Dataset.zip"
        if not archive.exists():
            download_archive(archive)
    if not archive.exists():
        raise SystemExit(f"dataset archive not found: {archive}")
    digest = md5sum(archive)
    if digest != SOURCE_ARCHIVE_MD5:
        raise SystemExit(f"archive checksum mismatch: expected {SOURCE_ARCHIVE_MD5}, got {digest}")

    records = normalize(archive, output)
    metadata_path = write_metadata(output, archive, records)
    print(f"Normalized {records:,} real sensor records")
    print(f"Dataset: {output.resolve()}")
    print(f"Metadata: {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
