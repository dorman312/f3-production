import logging
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from state import store
from state import app_state

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/venues", tags=["cameras"])


class CameraRegister(BaseModel):
    camera_id: str
    rtsp_url: str
    zone_id: str
    name: str = ""
    coverage_pct: float = 100.0


class FramePayload(BaseModel):
    frame_b64: str


@router.post("/{venue_id}/cameras", status_code=201)
def register_camera(venue_id: str, body: CameraRegister) -> dict:
    if not store.get_venue(venue_id):
        raise HTTPException(status_code=404, detail="Venue not found")
    if not store.get_zone(venue_id, body.zone_id):
        raise HTTPException(status_code=404, detail="Zone not found")
    if app_state.camera_manager is None:
        raise HTTPException(status_code=503, detail="Camera manager not initialized")
    return app_state.camera_manager.connect(
        camera_id=body.camera_id,
        rtsp_url=body.rtsp_url,
        zone_id=body.zone_id,
        venue_id=venue_id,
        name=body.name,
        coverage_pct=body.coverage_pct,
    )


@router.get("/{venue_id}/cameras")
def list_cameras(venue_id: str) -> list[dict]:
    if not store.get_venue(venue_id):
        raise HTTPException(status_code=404, detail="Venue not found")
    if app_state.camera_manager is None:
        return []
    return [c for c in app_state.camera_manager.get_status() if c["venue_id"] == venue_id]


@router.delete("/{venue_id}/cameras/{camera_id}", status_code=200)
def remove_camera(venue_id: str, camera_id: str) -> dict:
    if not store.get_venue(venue_id):
        raise HTTPException(status_code=404, detail="Venue not found")
    if app_state.camera_manager is None:
        raise HTTPException(status_code=503, detail="Camera manager not initialized")
    if not app_state.camera_manager.disconnect(camera_id):
        raise HTTPException(status_code=404, detail="Camera not found")
    return {"message": "Camera disconnected", "camera_id": camera_id}


@router.post("/{venue_id}/cameras/{camera_id}/frame")
def push_frame(venue_id: str, camera_id: str, body: FramePayload) -> dict:
    if not store.get_venue(venue_id):
        raise HTTPException(status_code=404, detail="Venue not found")
    if app_state.camera_manager is None:
        raise HTTPException(status_code=503, detail="Camera manager not initialized")

    result = app_state.camera_manager.process_frame(camera_id, body.frame_b64)
    if result is None:
        raise HTTPException(
            status_code=404, detail="Camera not found or frame decode failed"
        )

    wait_minutes = 0.0
    zone = store.get_zone(venue_id, result["zone_id"])
    if zone and app_state.predictor is not None:
        hour = datetime.now(timezone.utc).hour
        wait_minutes = app_state.predictor.predict(
            zone.type, result["density_score"], hour
        )
    elif zone:
        from state.store import get_wait_factor
        wait_minutes = round(result["density_score"] * get_wait_factor(zone.type), 1)

    return {
        "person_count": result["person_count"],
        "density_score": result["density_score"],
        "wait_minutes": round(wait_minutes, 1),
        "zone_coverage_pct": result.get("zone_coverage_pct", 0.0),
        "camera_id": camera_id,
        "zone_id": result["zone_id"],
        "timestamp": result["timestamp"],
    }
