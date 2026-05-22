from fastapi import APIRouter, HTTPException
from models.venue import Venue, VenueCreate, VenueWithZones
from state import store

router = APIRouter(prefix="/venues", tags=["venues"])


@router.post("", response_model=Venue, status_code=201)
def create_venue(body: VenueCreate) -> Venue:
    venue = Venue(**body.model_dump())
    store.venues[venue.id] = venue
    store.zones[venue.id] = {}
    store.alerts[venue.id] = {}
    return venue


@router.get("", response_model=list[Venue])
def list_venues() -> list[Venue]:
    return list(store.venues.values())


@router.get("/{venue_id}", response_model=VenueWithZones)
def get_venue(venue_id: str) -> VenueWithZones:
    venue = store.get_venue(venue_id)
    if not venue:
        raise HTTPException(status_code=404, detail="Venue not found")
    zones = list(store.get_zones(venue_id).values())
    return VenueWithZones(
        id=venue.id,
        name=venue.name,
        type=venue.type,
        address=venue.address,
        created_at=venue.created_at,
        zones=zones,
    )


@router.delete("/{venue_id}", status_code=200)
def delete_venue(venue_id: str) -> dict:
    if venue_id not in store.venues:
        raise HTTPException(status_code=404, detail="Venue not found")
    del store.venues[venue_id]
    store.zones.pop(venue_id, None)
    store.alerts.pop(venue_id, None)
    store.readings[:] = [r for r in store.readings if r.venue_id != venue_id]
    return {"message": "Venue deleted", "venue_id": venue_id}
