import logging
from datetime import datetime, timezone
from models.venue import Alert, DensityReading, Venue, Zone

logger = logging.getLogger(__name__)

venues: dict[str, Venue] = {}
zones: dict[str, dict[str, Zone]] = {}
readings: list[DensityReading] = []
alerts: dict[str, dict[str, Alert]] = {}

ZONE_WAIT_FACTORS: dict[str, float] = {
    "concession": 0.15,
    "concessions": 0.15,
    "restroom": 0.12,
    "restrooms": 0.12,
    "bathroom": 0.12,
    "bathrooms": 0.12,
    "gate": 0.08,
    "gates": 0.08,
    "exit": 0.10,
    "exits": 0.10,
}


def get_wait_factor(zone_type: str) -> float:
    return ZONE_WAIT_FACTORS.get(zone_type.lower(), 0.10)


def get_venue(venue_id: str) -> Venue | None:
    return venues.get(venue_id)


def get_zones(venue_id: str) -> dict[str, Zone]:
    return zones.get(venue_id, {})


def get_zone(venue_id: str, zone_id: str) -> Zone | None:
    return zones.get(venue_id, {}).get(zone_id)


def get_latest_reading(venue_id: str, zone_id: str) -> DensityReading | None:
    matches = [
        r for r in readings
        if r.venue_id == venue_id and r.zone_id == zone_id
    ]
    return max(matches, key=lambda r: r.timestamp, default=None)


def calculate_density(venue_id: str, zone_id: str) -> tuple[float, int]:
    zone = get_zone(venue_id, zone_id)
    if not zone:
        return 0.0, 0
    reading = get_latest_reading(venue_id, zone_id)
    if not reading:
        return 0.0, 0
    if zone.max_capacity <= 0:
        return 0.0, reading.person_count
    score = min(100.0, (reading.person_count / zone.max_capacity) * 100.0)
    return round(score, 2), reading.person_count


def get_risk_level(density_score: float) -> str:
    if density_score >= 90:
        return "high"
    if density_score >= 75:
        return "medium"
    return "low"


def add_reading(reading: DensityReading) -> None:
    readings.append(reading)
    if len(readings) > 10000:
        readings.pop(0)
    logger.info(
        "reading added venue=%s zone=%s count=%d source=%s",
        reading.venue_id, reading.zone_id, reading.person_count, reading.source,
    )
    generate_alerts(reading.venue_id)


def generate_alerts(venue_id: str) -> None:
    zone_map = zones.get(venue_id, {})
    venue_alerts = alerts.setdefault(venue_id, {})

    for zone in zone_map.values():
        density_score, _ = calculate_density(venue_id, zone.id)

        active = next(
            (a for a in venue_alerts.values() if a.zone_id == zone.id and not a.resolved),
            None,
        )

        if active:
            if density_score < 75:
                active.resolved = True
                active.resolved_at = datetime.now(timezone.utc)
                logger.info("alert auto-resolved zone=%s", zone.name)
            else:
                active.density_score = density_score
                active.level = get_risk_level(density_score)
                active.message = (
                    f"Zone '{zone.name}' has {active.level} congestion "
                    f"({density_score:.1f}% capacity)"
                )
            continue

        if density_score < 75:
            continue

        level = get_risk_level(density_score)
        alert = Alert(
            venue_id=venue_id,
            zone_id=zone.id,
            zone_name=zone.name,
            level=level,
            density_score=density_score,
            message=(
                f"Zone '{zone.name}' has {level} congestion "
                f"({density_score:.1f}% capacity)"
            ),
        )
        venue_alerts[alert.id] = alert
        logger.warning(
            "alert created zone=%s level=%s score=%.1f", zone.name, level, density_score
        )
