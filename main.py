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
