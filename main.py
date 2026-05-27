import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from routers import (
    alerts, analytics, cameras, congestion, ingest,
    map, predictions, routing, venues, waittime, zones,
)
from state import app_state

load_dotenv()

_level = logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO
logging.basicConfig(
    level=_level,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger(__name__)


# ── Demo seed ─────────────────────────────────────────────────────────────────

def _seed_demo_data() -> None:
    """
    Creates a fixed-ID venue with zones and crowd readings if the store is empty.

    The venue ID is intentionally hardcoded so the operator dashboard and any
    other clients that embed the ID stay consistent across server restarts.
    Called once at the end of the lifespan startup block.
    """
    from state import store
    from models.venue import Venue, Zone, DensityReading

    if store.venues:
        logger.info("seed: store already has %d venue(s) — skipping", len(store.venues))
        return

    # ── Venue ──────────────────────────────────────────────────────
    VENUE_ID = "52e70b13-b7c2-47ab-ad00-97a16c2f85dd"

    venue = Venue(
        id=VENUE_ID,
        name="US Bank Stadium",
        type="stadium",
        address="401 Chicago Ave, Minneapolis, MN",
    )
    store.venues[venue.id] = venue
    store.zones[venue.id]  = {}
    store.alerts[venue.id] = {}

    # ── Zones ───────────────────────────────────────────────────────
    # Each tuple: (name, type, x_pct, y_pct, w_pct, h_pct, area_m2, max_capacity)
    _ZONE_DEFS = [
        ("Gate A",             "gate",       0.10, 0.05, 0.15, 0.10, 200, 500),
        ("Gate B",             "gate",       0.75, 0.05, 0.15, 0.10, 200, 500),
        ("Concession Stand 1", "concession", 0.20, 0.35, 0.20, 0.12, 150, 300),
        ("Concession Stand 2", "concession", 0.60, 0.35, 0.20, 0.12, 150, 300),
        ("Concession Stand 3", "concession", 0.40, 0.50, 0.20, 0.12, 150, 300),
        ("Restroom Level 1",   "restroom",   0.10, 0.60, 0.15, 0.10, 100, 200),
        ("Restroom Level 2",   "restroom",   0.75, 0.60, 0.15, 0.10, 100, 200),
        ("Main Exit North",    "exit",       0.35, 0.85, 0.15, 0.10, 300, 800),
        ("Main Exit South",    "exit",       0.55, 0.85, 0.15, 0.10, 300, 800),
    ]

    # ── Crowd counts ────────────────────────────────────────────────
    _CROWD = {
        "Gate A":             142,
        "Gate B":              89,
        "Concession Stand 1":  63,
        "Concession Stand 2": 211,
        "Concession Stand 3":  95,
        "Restroom Level 1":    28,
        "Restroom Level 2":    77,
        "Main Exit North":     34,
        "Main Exit South":    156,
    }

    now = datetime.now(timezone.utc)

    for name, ztype, x, y, w, h, area, cap in _ZONE_DEFS:
        zone = Zone(
            venue_id=VENUE_ID,
            name=name,
            type=ztype,
            x_pct=x,
            y_pct=y,
            w_pct=w,
            h_pct=h,
            area_m2=float(area),
            max_capacity=cap,
        )
        store.zones[VENUE_ID][zone.id] = zone

        reading = DensityReading(
            venue_id=VENUE_ID,
            zone_id=zone.id,
            person_count=_CROWD[name],
            timestamp=now,
            source="demo",
        )
        store.readings.append(reading)

    # Alert generation runs automatically via add_reading; trigger it once
    # now that all readings are loaded so high-density zones get alerts.
    store.generate_alerts(VENUE_ID)

    zone_count = len(store.zones[VENUE_ID])
    logger.info(
        "seed: created venue '%s'  id=%s  zones=%d",
        venue.name, VENUE_ID, zone_count,
    )
    logger.info("seed: venue ID for dashboard → %s", VENUE_ID)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("F3 starting — initialising all services")

    try:
        from ml.predictor import WaitTimePredictor
        predictor = WaitTimePredictor()
        predictor.train()
        app_state.predictor = predictor
        logger.info("F3 ML predictor ready")
    except Exception as exc:
        logger.error("ML predictor init failed (falling back to formula): %s", exc)
        app_state.predictor = None

    try:
        from services.camera_service import RTSPCameraManager
        app_state.camera_manager = RTSPCameraManager()
        logger.info("F3 camera manager ready")
    except Exception as exc:
        logger.error("Camera manager init failed: %s", exc)
        app_state.camera_manager = None

    try:
        from services.map_service import VenueMapService
        app_state.map_service = VenueMapService()
        logger.info("F3 map service ready")
    except Exception as exc:
        logger.error("Map service init failed: %s", exc)
        app_state.map_service = None

    try:
        from services.routing_service import SmartRoutingService
        app_state.routing_service = SmartRoutingService()
        logger.info("F3 routing service ready")
    except Exception as exc:
        logger.error("Routing service init failed: %s", exc)
        app_state.routing_service = None

    try:
        from services.prediction_service import VenuePredictionService
        app_state.prediction_service = VenuePredictionService()
        logger.info("F3 prediction service ready")
    except Exception as exc:
        logger.error("Prediction service init failed: %s", exc)
        app_state.prediction_service = None

    stream_processor = None
    try:
        from workers.stream_processor import StreamProcessor
        ingest_base = os.getenv("INGEST_BASE_URL", "http://localhost:8000")
        stream_processor = StreamProcessor(ingest_base=ingest_base)
        asyncio.create_task(stream_processor.start())
        logger.info("F3 stream processor started (ingest base: %s)", ingest_base)
    except Exception as exc:
        logger.error("Stream processor init failed: %s", exc)

    _seed_demo_data()
    logger.info("F3 ready — all components online")
    yield
    logger.info("F3 shutting down")
    if stream_processor is not None:
        await stream_processor.stop()


app = FastAPI(
    title="Fan Flow & Fusion (F3)",
    description=(
        "Universal venue crowd intelligence platform. "
        "Works in any venue: stadiums, malls, airports, arenas, hospitals, train stations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(venues.router)
app.include_router(zones.router)
app.include_router(ingest.router)
app.include_router(congestion.router)
app.include_router(waittime.router)
app.include_router(routing.router)
app.include_router(alerts.router)
app.include_router(analytics.router)
app.include_router(cameras.router)
app.include_router(map.router)
app.include_router(predictions.router)


@app.get("/", tags=["health"])
def root() -> dict:
    return {
        "service": "Fan Flow & Fusion (F3)",
        "description": "Universal venue crowd intelligence platform",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["health"])
def health() -> dict:
    from state import store
    predictor = app_state.predictor
    manager = app_state.camera_manager
    return {
        "status": "healthy",
        "venues": len(store.venues),
        "zones": sum(len(v) for v in store.zones.values()),
        "readings": len(store.readings),
        "active_alerts": sum(
            sum(1 for a in v.values() if not a.resolved)
            for v in store.alerts.values()
        ),
        "ml_predictor_trained": predictor is not None and getattr(predictor, "_trained", False),
        "cameras_registered": manager.camera_count if manager is not None else 0,
        "services": {
            "map": app_state.map_service is not None,
            "routing": app_state.routing_service is not None,
            "prediction": app_state.prediction_service is not None,
        },
    }


@app.get("/venues/{venue_id}/live", tags=["live"])
def get_live(venue_id: str) -> dict:
    from state import store
    if not store.get_venue(venue_id):
        raise HTTPException(status_code=404, detail="Venue not found")

    venue_map: dict | None = None
    if app_state.map_service is not None:
        venue_map = app_state.map_service.generate_map(venue_id)

    forecast: dict | None = None
    if app_state.prediction_service is not None:
        forecast = app_state.prediction_service.predict_next_30min(venue_id)

    active_alerts = [
        a.model_dump()
        for a in store.alerts.get(venue_id, {}).values()
        if not a.resolved
    ]

    return {
        "venue_id": venue_id,
        "map": venue_map,
        "predictions": forecast,
        "alerts": active_alerts,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
