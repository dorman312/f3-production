from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from models.venue import AnalyticsSummary, ZoneDensity
from state import store

router = APIRouter(prefix="/venues", tags=["analytics"])


@router.get("/{venue_id}/analytics/summary", response_model=AnalyticsSummary)
def get_analytics_summary(venue_id: str) -> AnalyticsSummary:
    if not store.get_venue(venue_id):
        raise HTTPException(status_code=404, detail="Venue not found")

    venue = store.get_venue(venue_id)
    zone_map = store.get_zones(venue_id)
    today = datetime.now(timezone.utc).date()

    today_readings = [
        r for r in store.readings
        if r.venue_id == venue_id and r.timestamp.date() == today
    ]

    zone_densities: list[ZoneDensity] = []
    for zone in zone_map.values():
        density_score, person_count = store.calculate_density(venue_id, zone.id)
        zone_densities.append(ZoneDensity(
            zone_id=zone.id,
            zone_name=zone.name,
            zone_type=zone.type,
            density_score=density_score,
            risk_level=store.get_risk_level(density_score),
            person_count=person_count,
            wait_minutes=round(density_score * store.get_wait_factor(zone.type), 1),
            max_capacity=zone.max_capacity,
        ))

    zone_densities.sort(key=lambda z: z.density_score, reverse=True)

    avg_density = (
        sum(z.density_score for z in zone_densities) / len(zone_densities)
        if zone_densities else 0.0
    )

    active_alerts = sum(
        1 for a in store.alerts.get(venue_id, {}).values() if not a.resolved
    )

    return AnalyticsSummary(
        venue_id=venue_id,
        venue_name=venue.name,
        venue_type=venue.type,
        total_zones=len(zone_map),
        total_readings_today=len(today_readings),
        average_density=round(avg_density, 1),
        active_alerts=active_alerts,
        peak_zones=zone_densities[:5],
    )
