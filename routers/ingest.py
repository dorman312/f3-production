from fastapi import APIRouter, HTTPException
from models.venue import CameraIngest, DensityReading, ManualIngest, WifiIngest
from state import store

router = APIRouter(prefix="/ingest", tags=["ingest"])


def _require_zone(venue_id: str, zone_id: str) -> None:
    if not store.get_venue(venue_id):
        raise HTTPException(status_code=404, detail="Venue not found")
    if zone_id not in store.get_zones(venue_id):
        raise HTTPException(status_code=404, detail="Zone not found")


@router.post("/camera", status_code=201)
def ingest_camera(body: CameraIngest) -> dict:
    _require_zone(body.venue_id, body.zone_id)
    reading = DensityReading(
        venue_id=body.venue_id,
        zone_id=body.zone_id,
        person_count=body.person_count,
        timestamp=body.timestamp,
        source="camera",
        camera_id=body.camera_id,
    )
    store.add_reading(reading)
    return {"status": "accepted", "reading_id": reading.id}


@router.post("/wifi", status_code=201)
def ingest_wifi(body: WifiIngest) -> dict:
    _require_zone(body.venue_id, body.zone_id)
    person_count = max(0, int(body.device_count / 1.1))
    reading = DensityReading(
        venue_id=body.venue_id,
        zone_id=body.zone_id,
        person_count=person_count,
        timestamp=body.timestamp,
        source="wifi",
    )
    store.add_reading(reading)
    return {"status": "accepted", "reading_id": reading.id}


@router.post("/manual", status_code=201)
def ingest_manual(body: ManualIngest) -> dict:
    _require_zone(body.venue_id, body.zone_id)
    reading = DensityReading(
        venue_id=body.venue_id,
        zone_id=body.zone_id,
        person_count=body.person_count,
        timestamp=body.timestamp,
        source="manual",
    )
    store.add_reading(reading)
    return {"status": "accepted", "reading_id": reading.id}
