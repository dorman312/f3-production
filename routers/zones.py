from fastapi import APIRouter, HTTPException
from models.venue import Zone, ZoneCreate, ZoneUpdate
from state import store

router = APIRouter(prefix="/venues", tags=["zones"])


@router.post("/{venue_id}/zones", response_model=Zone, status_code=201)
def create_zone(venue_id: str, body: ZoneCreate) -> Zone:
    if not store.get_venue(venue_id):
        raise HTTPException(status_code=404, detail="Venue not found")
    zone = Zone(venue_id=venue_id, **body.model_dump())
    store.zones[venue_id][zone.id] = zone
    return zone


@router.get("/{venue_id}/zones", response_model=list[Zone])
def list_zones(venue_id: str) -> list[Zone]:
    if not store.get_venue(venue_id):
        raise HTTPException(status_code=404, detail="Venue not found")
    return list(store.get_zones(venue_id).values())


@router.put("/{venue_id}/zones/{zone_id}", response_model=Zone)
def update_zone(venue_id: str, zone_id: str, body: ZoneUpdate) -> Zone:
    if not store.get_venue(venue_id):
        raise HTTPException(status_code=404, detail="Venue not found")
    zone = store.get_zone(venue_id, zone_id)
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")
    updates = body.model_dump(exclude_unset=True)
    updated = zone.model_copy(update=updates)
    store.zones[venue_id][zone_id] = updated
    return updated
