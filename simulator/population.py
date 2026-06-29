import random
from dataclasses import dataclass

from simulator.models import PopulationSnapshot, PreferenceProfile


@dataclass(frozen=True)
class OccupantArchetype:
    name: str
    weight: float
    preferred_temperature: float
    preferred_humidity: float
    preferred_light: float
    energy_sensitivity: float
    discomfort_tolerance: float
    crowding_sensitivity: float
    activity_level: float
    weekday_arrival: int
    weekday_departure: int
    weekend_arrival: int | None
    weekend_departure: int | None
    presence_probability: float
    weekend_probability: float
    schedule_jitter_minutes: int


@dataclass(frozen=True)
class SyntheticOccupant:
    occupant_id: int
    archetype: OccupantArchetype
    preferred_temperature: float
    preferred_humidity: float
    preferred_light: float
    energy_sensitivity: float
    discomfort_tolerance: float
    crowding_sensitivity: float
    activity_level: float
    routine_offset_minutes: int
    absenteeism: float

    def is_present(self, day: int, minute: int, seed: int) -> bool:
        is_weekend = day % 7 in (5, 6)
        if is_weekend and self.archetype.weekend_arrival is None:
            return False

        day_rng = random.Random(seed + self.occupant_id * 1009 + day * 9176)
        probability = self.archetype.weekend_probability if is_weekend else self.archetype.presence_probability
        if day_rng.random() > probability * (1.0 - self.absenteeism):
            return False

        arrival_base = self.archetype.weekend_arrival if is_weekend else self.archetype.weekday_arrival
        departure_base = self.archetype.weekend_departure if is_weekend else self.archetype.weekday_departure
        if arrival_base is None or departure_base is None:
            return False

        jitter = day_rng.randint(-self.archetype.schedule_jitter_minutes, self.archetype.schedule_jitter_minutes)
        lunch_shift = day_rng.randint(-15, 15)
        arrival = arrival_base + self.routine_offset_minutes + jitter
        departure = departure_base + self.routine_offset_minutes + jitter

        if not arrival <= minute < departure:
            return False

        if not is_weekend and 12 * 60 + lunch_shift <= minute < 13 * 60 + lunch_shift:
            return day_rng.random() < 0.35

        return True

    def preference_profile(self) -> PreferenceProfile:
        return PreferenceProfile(
            target_temperature=self.preferred_temperature,
            target_humidity=self.preferred_humidity,
            target_light_when_occupied=self.preferred_light,
            energy_sensitivity=self.energy_sensitivity,
            discomfort_tolerance=self.discomfort_tolerance,
            crowding_sensitivity=self.crowding_sensitivity,
            activity_level=self.activity_level,
        )


ARCHETYPES = (
    OccupantArchetype(
        name="office_regular",
        weight=0.38,
        preferred_temperature=24.8,
        preferred_humidity=49.0,
        preferred_light=520.0,
        energy_sensitivity=0.85,
        discomfort_tolerance=1.0,
        crowding_sensitivity=1.0,
        activity_level=0.95,
        weekday_arrival=9 * 60,
        weekday_departure=17 * 60 + 30,
        weekend_arrival=None,
        weekend_departure=None,
        presence_probability=0.92,
        weekend_probability=0.0,
        schedule_jitter_minutes=28,
    ),
    OccupantArchetype(
        name="early_bird",
        weight=0.16,
        preferred_temperature=24.2,
        preferred_humidity=47.0,
        preferred_light=560.0,
        energy_sensitivity=0.75,
        discomfort_tolerance=0.9,
        crowding_sensitivity=0.9,
        activity_level=1.05,
        weekday_arrival=7 * 60 + 30,
        weekday_departure=15 * 60 + 45,
        weekend_arrival=None,
        weekend_departure=None,
        presence_probability=0.88,
        weekend_probability=0.0,
        schedule_jitter_minutes=24,
    ),
    OccupantArchetype(
        name="late_collaborator",
        weight=0.16,
        preferred_temperature=25.7,
        preferred_humidity=52.0,
        preferred_light=470.0,
        energy_sensitivity=0.95,
        discomfort_tolerance=1.15,
        crowding_sensitivity=1.1,
        activity_level=1.0,
        weekday_arrival=10 * 60 + 15,
        weekday_departure=19 * 60,
        weekend_arrival=None,
        weekend_departure=None,
        presence_probability=0.82,
        weekend_probability=0.0,
        schedule_jitter_minutes=35,
    ),
    OccupantArchetype(
        name="evening_user",
        weight=0.12,
        preferred_temperature=25.3,
        preferred_humidity=51.0,
        preferred_light=430.0,
        energy_sensitivity=1.1,
        discomfort_tolerance=1.2,
        crowding_sensitivity=0.85,
        activity_level=0.8,
        weekday_arrival=17 * 60,
        weekday_departure=21 * 60 + 30,
        weekend_arrival=16 * 60,
        weekend_departure=20 * 60,
        presence_probability=0.55,
        weekend_probability=0.32,
        schedule_jitter_minutes=42,
    ),
    OccupantArchetype(
        name="energy_saver",
        weight=0.10,
        preferred_temperature=26.4,
        preferred_humidity=54.0,
        preferred_light=390.0,
        energy_sensitivity=1.55,
        discomfort_tolerance=1.35,
        crowding_sensitivity=0.8,
        activity_level=0.9,
        weekday_arrival=9 * 60 + 20,
        weekday_departure=17 * 60,
        weekend_arrival=None,
        weekend_departure=None,
        presence_probability=0.78,
        weekend_probability=0.0,
        schedule_jitter_minutes=30,
    ),
    OccupantArchetype(
        name="comfort_seeker",
        weight=0.08,
        preferred_temperature=23.7,
        preferred_humidity=46.0,
        preferred_light=610.0,
        energy_sensitivity=0.55,
        discomfort_tolerance=0.7,
        crowding_sensitivity=1.25,
        activity_level=1.1,
        weekday_arrival=8 * 60 + 45,
        weekday_departure=18 * 60,
        weekend_arrival=10 * 60,
        weekend_departure=14 * 60,
        presence_probability=0.86,
        weekend_probability=0.16,
        schedule_jitter_minutes=26,
    ),
)


class SyntheticPopulation:
    def __init__(self, occupants: list[SyntheticOccupant], seed: int = 0):
        self.occupants = occupants
        self.seed = seed

    @classmethod
    def generate(cls, size: int = 24, seed: int | None = None) -> "SyntheticPopulation":
        rng = random.Random(seed)
        occupants = []
        weights = [archetype.weight for archetype in ARCHETYPES]
        for occupant_id in range(size):
            archetype = rng.choices(ARCHETYPES, weights=weights, k=1)[0]
            occupants.append(
                SyntheticOccupant(
                    occupant_id=occupant_id,
                    archetype=archetype,
                    preferred_temperature=rng.gauss(archetype.preferred_temperature, 0.45),
                    preferred_humidity=rng.gauss(archetype.preferred_humidity, 3.0),
                    preferred_light=max(280.0, rng.gauss(archetype.preferred_light, 55.0)),
                    energy_sensitivity=max(0.35, rng.gauss(archetype.energy_sensitivity, 0.12)),
                    discomfort_tolerance=max(0.45, rng.gauss(archetype.discomfort_tolerance, 0.12)),
                    crowding_sensitivity=max(0.45, rng.gauss(archetype.crowding_sensitivity, 0.12)),
                    activity_level=max(0.45, rng.gauss(archetype.activity_level, 0.15)),
                    routine_offset_minutes=rng.randint(-18, 18),
                    absenteeism=rng.uniform(0.0, 0.08),
                )
            )
        return cls(occupants=occupants, seed=seed or 0)

    def snapshot(self, day: int, minute: int) -> PopulationSnapshot:
        active = [
            occupant
            for occupant in self.occupants
            if occupant.is_present(day, minute, self.seed)
        ]
        counts: dict[str, int] = {}
        for occupant in active:
            counts[occupant.archetype.name] = counts.get(occupant.archetype.name, 0) + 1

        return PopulationSnapshot(
            day=day,
            minute=minute,
            is_weekend=day % 7 in (5, 6),
            active_occupant_ids=tuple(occupant.occupant_id for occupant in active),
            active_archetypes=counts,
            preferences=self._aggregate_preferences(active),
        )

    def active_occupants(self, snapshot: PopulationSnapshot) -> list[SyntheticOccupant]:
        active_ids = set(snapshot.active_occupant_ids)
        return [occupant for occupant in self.occupants if occupant.occupant_id in active_ids]

    def _aggregate_preferences(self, active: list[SyntheticOccupant]) -> PreferenceProfile:
        if not active:
            return PreferenceProfile()

        count = len(active)
        return PreferenceProfile(
            target_temperature=sum(occupant.preferred_temperature for occupant in active) / count,
            target_humidity=sum(occupant.preferred_humidity for occupant in active) / count,
            target_light_when_occupied=sum(occupant.preferred_light for occupant in active) / count,
            energy_sensitivity=sum(occupant.energy_sensitivity for occupant in active) / count,
            discomfort_tolerance=sum(occupant.discomfort_tolerance for occupant in active) / count,
            crowding_sensitivity=sum(occupant.crowding_sensitivity for occupant in active) / count,
            activity_level=sum(occupant.activity_level for occupant in active) / count,
        )
