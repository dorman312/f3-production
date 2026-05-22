import logging
from fastapi import APIRouter, HTTPException
from state import app_state
from state import store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/venues", tags=["map"])


@router.get("/{venue_id}/map/zone/{zone_id}")
def get_zone_detail(venue_id: str, zone_id: str) -> dict:
    if not store.get_venue(venue_id):
        raise HTTPException(status_code=404, detail="Venue not found")
    if app_state.map_service is None:
        raise HTTPException(status_code=503, detail="Map service not initialized")
    result = app_state.map_service.get_zone_detail(venue_id, zone_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Zone not found")
    return result


@router.get("/{venue_id}/map")
def get_venue_map(venue_id: str) -> dict:
    if not store.get_venue(venue_id):
        raise HTTPException(status_code=404, detail="Venue not found")
    if app_state.map_service is None:
        raise HTTPException(status_code=503, detail="Map service not initialized")
    result = app_state.map_service.generate_map(venue_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Venue not found")
    return result
