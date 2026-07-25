from __future__ import annotations

import csv
import gzip
import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

from room_control.models import ControlAction, RoomObservation


DATASET_TITLE = "Room-level occupant counts and environmental quality from heterogeneous sensing modalities in a smart building"
DATASET_DOI = "https://doi.org/10.6084/m9.figshare.9959417.v1"
PAPER_DOI = "https://doi.org/10.1038/s41597-019-0274-4"
SOURCE_ARCHIVE_URL = "https://ndownloader.figshare.com/files/17938001"
SOURCE_ARCHIVE_MD5 = "3466bca7aae4c8f1118731dbdf162110"

NORMALIZED_FIELDS = (
    "room_id",
    "room_type",
    "area_m2",
    "volume_m3",
    "seating_capacity",
    "year",
    "month",
    "day_id",
    "minute_of_day",
    "is_workday",
    "occupancy",
    "temperature_c",
    "humidity_pct",
    "co2_ppm",
    "illuminance_lux",
    "vav_damper_pct",
)


@dataclass(frozen=True)
class DatasetSummary:
    records: int
    rooms: tuple[str, ...]
    day_ids: tuple[int, ...]
    occupied_records: int
    maximum_occupancy: int
    average_temperature_c: float
    average_humidity_pct: float
    average_co2_ppm: float
    average_illuminance_lux: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def iter_observations(
    path: str | Path,
    rooms: set[str] | None = None,
) -> Iterator[RoomObservation]:
    source_path = Path(path)
    previous_actions: dict[str, ControlAction] = {}
    with _open_text(source_path) as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != NORMALIZED_FIELDS:
            raise ValueError(
                f"unexpected dataset schema in {source_path}; expected {NORMALIZED_FIELDS}"
            )
        for row_number, row in enumerate(reader, start=2):
            room_id = row["room_id"]
            if rooms is not None and room_id not in rooms:
                continue
            try:
                observation = RoomObservation(
                    room_id=room_id,
                    room_type=row["room_type"],
                    area_m2=float(row["area_m2"]),
                    volume_m3=float(row["volume_m3"]),
                    seating_capacity=int(row["seating_capacity"]),
                    year=int(row["year"]),
                    month=int(row["month"]),
                    day_id=int(row["day_id"]),
                    minute_of_day=int(row["minute_of_day"]),
                    is_workday=row["is_workday"].strip().lower() in {"1", "true", "yes"},
                    occupancy=max(0, int(round(float(row["occupancy"])))),
                    temperature_c=float(row["temperature_c"]),
                    humidity_pct=float(row["humidity_pct"]),
                    co2_ppm=float(row["co2_ppm"]),
                    illuminance_lux=float(row["illuminance_lux"]),
                    logged_ventilation_pct=float(row["vav_damper_pct"]),
                    previous_action=previous_actions.get(room_id, ControlAction()),
                    source="sdu_real_room_dataset",
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid normalized record at row {row_number}: {exc}") from exc
            yield observation


def load_observations(
    path: str | Path,
    rooms: set[str] | None = None,
) -> list[RoomObservation]:
    return list(iter_observations(path, rooms=rooms))


def split_day_ids(
    observations: Iterable[RoomObservation],
    train_fraction: float = 0.8,
    seed: int = 42,
) -> tuple[set[int], set[int]]:
    if not 0.1 <= train_fraction <= 0.9:
        raise ValueError("train_fraction must be between 0.1 and 0.9")
    day_ids = sorted({observation.day_id for observation in observations})
    if len(day_ids) < 2:
        raise ValueError("at least two whole days are required for a split")
    random.Random(seed).shuffle(day_ids)
    split_index = max(1, min(len(day_ids) - 1, round(len(day_ids) * train_fraction)))
    return set(day_ids[:split_index]), set(day_ids[split_index:])


def split_train_validation_test_day_ids(
    observations: Iterable[RoomObservation],
    development_fraction: float = 0.8,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[set[int], set[int], set[int]]:
    """Split whole days into train, validation, and untouched final test sets.

    ``validation_fraction`` is the share of the development set, so the defaults
    produce 28 train, 7 validation, and 9 test days for the 44-day dataset.
    """
    if not 0.5 <= development_fraction <= 0.9:
        raise ValueError("development_fraction must be between 0.5 and 0.9")
    if not 0.1 <= validation_fraction <= 0.4:
        raise ValueError("validation_fraction must be between 0.1 and 0.4")
    day_ids = sorted({observation.day_id for observation in observations})
    if len(day_ids) < 3:
        raise ValueError("at least three whole days are required for a three-way split")
    random.Random(seed).shuffle(day_ids)
    development_count = max(
        2,
        min(len(day_ids) - 1, round(len(day_ids) * development_fraction)),
    )
    development = day_ids[:development_count]
    test = day_ids[development_count:]
    validation_count = max(
        1,
        min(len(development) - 1, round(len(development) * validation_fraction)),
    )
    training_count = len(development) - validation_count
    return (
        set(development[:training_count]),
        set(development[training_count:]),
        set(test),
    )


def observations_for_days(
    observations: Iterable[RoomObservation],
    day_ids: set[int],
) -> list[RoomObservation]:
    return [observation for observation in observations if observation.day_id in day_ids]


def iter_real_transitions(
    observations: Iterable[RoomObservation],
) -> Iterator[tuple[RoomObservation, RoomObservation]]:
    previous: dict[tuple[str, int], RoomObservation] = {}
    for observation in observations:
        key = (observation.room_id, observation.day_id)
        prior = previous.get(key)
        if prior is not None and observation.minute_of_day == prior.minute_of_day + 1:
            yield prior, observation
        previous[key] = observation


def summarize(observations: Iterable[RoomObservation]) -> DatasetSummary:
    count = 0
    occupied = 0
    maximum_occupancy = 0
    rooms: set[str] = set()
    days: set[int] = set()
    totals = Counter()
    for observation in observations:
        count += 1
        occupied += int(observation.occupancy > 0)
        maximum_occupancy = max(maximum_occupancy, observation.occupancy)
        rooms.add(observation.room_id)
        days.add(observation.day_id)
        totals["temperature"] += observation.temperature_c
        totals["humidity"] += observation.humidity_pct
        totals["co2"] += observation.co2_ppm
        totals["illuminance"] += observation.illuminance_lux
    if count == 0:
        raise ValueError("dataset contains no observations")
    return DatasetSummary(
        records=count,
        rooms=tuple(sorted(rooms)),
        day_ids=tuple(sorted(days)),
        occupied_records=occupied,
        maximum_occupancy=maximum_occupancy,
        average_temperature_c=totals["temperature"] / count,
        average_humidity_pct=totals["humidity"] / count,
        average_co2_ppm=totals["co2"] / count,
        average_illuminance_lux=totals["illuminance"] / count,
    )


def read_metadata(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
