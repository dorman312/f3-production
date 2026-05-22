from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Query
from models.venue import WaitTime, WaitTimePrediction
from state import store
from state import app_state

router = APIRouter(prefix="/venues", tags=["waittimes"])


@router.get("/{venue_id}/waittimes/predict", response_model=WaitTimePrediction)
def predict_waittime(
    venue_id: str,
    zone_type: str = Query(..., description="Zone type (gate, concession, restroom, exit, corridor)"),
    current_density: float = Query(..., ge=0.0, le=100.0, description="Current density score 0-100"),
    hour: int = Query(..., ge=0, le=23, description="Hour of day (0-23)"),
) -> WaitTimePrediction:
    if not store.get_venue(venue_id):
        raise HTTPException(status_code=404, detail="Venue not found")

    if app_state.predictor is not None:
        pred = app_state.predictor.predict_with_confidence(zone_type, current_density, hour)
        return WaitTimePrediction(
            zone_type=zone_type,
            current_density=current_density,
            hour=hour,
            predicted_wait_minutes=pred["predicted"],
            confidence_low=pred["confidence_low"],
            confidence_high=pred["confidence_high"],
            model_trained=pred["model_trained"],
        )

    factor = store.get_wait_factor(zone_type)
    wait = round(current_density * factor, 1)
    return WaitTimePrediction(
        zone_type=zone_type,
        current_density=current_density,
        hour=hour,
        predicted_wait_minutes=wait,
        confidence_low=round(wait * 0.8, 1),
        confidence_high=round(wait * 1.2, 1),
        model_trained=False,
    )


@router.get("/{venue_id}/waittimes", response_model=list[WaitTime])
def get_waittimes(venue_id: str) -> list[WaitTime]:
    if not store.get_venue(venue_id):
        raise HTTPException(status_code=404, detail="Venue not found")
    hour = datetime.now(timezone.utc).hour
    result: list[WaitTime] = []
    for zone in store.get_zones(venue_id).values():
        density_score, _ = store.calculate_density(venue_id, zone.id)
        if app_state.predictor is not None:
            wait_minutes = app_state.predictor.predict(zone.type, density_score, hour)
        else:
            wait_minutes = round(density_score * store.get_wait_factor(zone.type), 1)
        result.append(WaitTime(
            zone_id=zone.id,
            zone_name=zone.name,
            zone_type=zone.type,
            density_score=density_score,
            wait_minutes=round(wait_minutes, 1),
        ))
    return result
