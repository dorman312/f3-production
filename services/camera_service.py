import base64
import logging
from datetime import datetime, timezone

import cv2
import numpy as np

from ml.crowd_analyzer import CrowdDensityAnalyzer
from models.venue import DensityReading
from state import store

logger = logging.getLogger(__name__)


class RTSPCameraManager:
    def __init__(self) -> None:
        self._cameras: dict[str, dict] = {}
        self._analyzers: dict[str, CrowdDensityAnalyzer] = {}
        self._last_results: dict[str, dict] = {}

    @property
    def camera_count(self) -> int:
        return len(self._cameras)

    def connect(
        self,
        camera_id: str,
        rtsp_url: str,
        zone_id: str,
        venue_id: str,
        name: str = "",
        coverage_pct: float = 100.0,
    ) -> dict:
        self._cameras[camera_id] = {
            "camera_id": camera_id,
            "rtsp_url": rtsp_url,
            "zone_id": zone_id,
            "venue_id": venue_id,
            "name": name,
            "coverage_pct": coverage_pct,
            "status": "connected",
            "connected_at": datetime.now(timezone.utc).isoformat(),
        }
        self._analyzers[camera_id] = CrowdDensityAnalyzer()
        logger.info("camera registered id=%s zone=%s venue=%s", camera_id, zone_id, venue_id)
        return dict(self._cameras[camera_id])

    def disconnect(self, camera_id: str) -> bool:
        if camera_id not in self._cameras:
            return False
        del self._cameras[camera_id]
        self._analyzers.pop(camera_id, None)
        self._last_results.pop(camera_id, None)
        logger.info("camera disconnected id=%s", camera_id)
        return True

    def get_status(self) -> list[dict]:
        result = []
        for cam_id, cam in self._cameras.items():
            entry = dict(cam)
            entry["last_analysis"] = self._last_results.get(cam_id)
            result.append(entry)
        return result

    def process_frame(self, camera_id: str, frame_b64: str) -> dict | None:
        cam = self._cameras.get(camera_id)
        if not cam:
            logger.warning("process_frame: unknown camera_id=%s", camera_id)
            return None

        try:
            img_bytes = base64.b64decode(frame_b64)
            buf = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("imdecode returned None — invalid image data")
        except Exception as exc:
            logger.error("frame decode failed camera=%s: %s", camera_id, exc)
            return None

        analyzer = self._analyzers.get(camera_id)
        if analyzer is None:
            logger.error("no analyzer for camera=%s", camera_id)
            return None

        analysis = analyzer.analyze_frame(frame)
        now = datetime.now(timezone.utc)

        result = {
            **analysis,
            "camera_id": camera_id,
            "zone_id": cam["zone_id"],
            "venue_id": cam["venue_id"],
            "timestamp": now.isoformat(),
        }
        self._last_results[camera_id] = result

        reading = DensityReading(
            venue_id=cam["venue_id"],
            zone_id=cam["zone_id"],
            person_count=analysis["person_count"],
            timestamp=now,
            source="camera",
            camera_id=camera_id,
        )
        store.add_reading(reading)

        return result
