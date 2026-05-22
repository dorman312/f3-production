import logging
from fastapi import APIRouter, HTTPException
from state import app_state
from state import store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/venues", tags=["predictions"])


@router.get("/{venue_id}/predictions/peak")
def get_peak_periods(venue_id: str) -> dict:
    if not store.get_venue(venue_id):
        raise HTTPException(status_code=404, detail="Venue not found")
    if app_state.prediction_service is None:
        raise HTTPException(status_code=503, detail="Prediction service not initialized")
    result = app_state.prediction_service.peak_periods(venue_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Venue not found")
    return result


@router.get("/{venue_id}/predictions")
def get_predictions(venue_id: str) -> dict:
    if not store.get_venue(venue_id):
        raise HTTPException(status_code=404, detail="Venue not found")
    if app_state.prediction_service is None:
        raise HTTPException(status_code=503, detail="Prediction service not initialized")
    result = app_state.prediction_service.predict_next_30min(venue_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Venue not found")
    return result
