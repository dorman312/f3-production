from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from models.venue import Zone, ZoneDensity
from state import store
from state import app_state

router = APIRouter(prefix="/venues", tags=["congestion"])


def _build_density(venue_id: str, zone: Zone) -> ZoneDensity:
    density_score, person_count = store.calculate_density(venue_id, zone.id)
    risk_level = store.get_risk_level(density_score)
    hour = datetime.now(timezone.utc).hour

    if app_state.predictor is not None:
        pred = app_state.predictor.predict_with_confidence(zone.type, density_score, hour)
        wait_minutes = pred["predicted"]
        interval = pred["confidence_high"] - pred["confidence_low"]
        confidence = round(max(0.0, 1.0 - interval / 20.0), 2)
    else:
        wait_minutes = round(density_score * store.get_wait_factor(zone.type), 1)
        confidence = 1.0

    return ZoneDensity(
        zone_id=zone.id,
        zone_name=zone.name,
        zone_type=zone.type,
        density_score=density_score,
        risk_level=risk_level,
        person_count=person_count,
        wait_minutes=round(wait_minutes, 1),
        max_capacity=zone.max_capacity,
        prediction_confidence=confidence,
    )


@router.get("/{venue_id}/congestion", response_model=list[ZoneDensity])
def get_congestion(venue_id: str) -> list[ZoneDensity]:
    if not store.get_venue(venue_id):
        raise HTTPException(status_code=404, detail="Venue not found")
    return [_build_density(venue_id, z) for z in store.get_zones(venue_id).values()]


@router.get("/{venue_id}/congestion/{zone_id}", response_model=ZoneDensity)
def get_zone_congestion(venue_id: str, zone_id: str) -> ZoneDensity:
    if not store.get_venue(venue_id):
        raise HTTPException(status_code=404, detail="Venue not found")
    zone = store.get_zone(venue_id, zone_id)
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    return _build_density(venue_id, zone)
