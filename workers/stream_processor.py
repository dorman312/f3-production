"""
workers/stream_processor.py

Background asyncio worker that polls every registered RTSP camera
once per FRAME_INTERVAL seconds, runs YOLOv8 person detection, and
posts accepted readings to POST /ingest/camera.

Readings are also written to a CSV log at logs/readings_YYYY-MM-DD.csv
so there is a persistent audit trail even though the in-memory store resets.

Integration with the FastAPI app (add to main.py lifespan):

    from workers.stream_processor import StreamProcessor
    processor = StreamProcessor()
    asyncio.create_task(processor.start())
"""

import asyncio
import csv
import json
import logging
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ml.crowd_analyzer import CrowdDensityAnalyzer
from state import app_state
from state import store

logger = logging.getLogger(__name__)

# ── Tuning constants ──────────────────────────────────────────────
FRAME_INTERVAL    = 15      # seconds between frame captures per camera
RECONNECT_INTERVAL= 30      # seconds between reconnect attempts after drop
SYNC_INTERVAL     = 30      # seconds between camera-registry sync passes
INITIAL_BACKOFF   = 5       # seconds before first reconnect attempt
MAX_BACKOFF       = 300     # cap reconnect wait at 5 minutes
INGEST_TIMEOUT    = 5       # seconds for the HTTP POST to /ingest/camera
MIN_CONFIDENCE    = 0.40    # frames with avg_confidence below this are skipped

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"


def _csv_path() -> Path:
    """One CSV file per calendar day (UTC)."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return _LOG_DIR / f"readings_{date_str}.csv"


_CSV_HEADER = ["timestamp", "venue_id", "zone_id", "zone_name",
               "camera_id", "person_count", "avg_confidence", "density_score"]


def _append_csv(row: dict) -> None:
    """Append one reading row to today's CSV. Creates file + header if needed."""
    path = _csv_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    try:
        with path.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=_CSV_HEADER)
            if write_header:
                w.writeheader()
            w.writerow(row)
    except OSError as exc:
        logger.warning("CSV write failed: %s", exc)


class StreamProcessor:
    """
    Manages one asyncio task per registered camera.

    Per-frame pipeline:
      1. Read one frame from the RTSP stream   (blocking → thread pool)
      2. Run CrowdDensityAnalyzer.analyze_frame()  (CPU → thread pool)
      3. If avg_confidence < MIN_CONFIDENCE, discard and log warning
      4. POST accepted reading to /ingest/camera
      5. Append reading to the daily CSV log

    A registry-watcher loop re-runs every SYNC_INTERVAL seconds so cameras
    registered or removed at runtime are picked up automatically.
    """

    def __init__(self, ingest_base: str = "http://localhost:8000") -> None:
        self._ingest_base = ingest_base.rstrip("/")
        self._tasks:     dict[str, asyncio.Task]         = {}
        self._analyzers: dict[str, CrowdDensityAnalyzer] = {}
        # One thread per camera for I/O + YOLO inference (YOLO not thread-safe;
        # each camera has its own analyzer instance so there is no contention)
        self._executor = ThreadPoolExecutor(
            max_workers=16, thread_name_prefix="rtsp-worker"
        )
        self._running = False

    # ── Lifecycle ─────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the registry watcher. Blocks until stop() is called."""
        self._running = True
        logger.info("StreamProcessor starting — ingest: %s  frame_interval: %ds  min_conf: %.2f",
                    self._ingest_base, FRAME_INTERVAL, MIN_CONFIDENCE)
        try:
            await self._watch_registry()
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Cancel all camera tasks and release resources."""
        self._running = False
        if self._tasks:
            logger.info("StreamProcessor stopping %d camera task(s)", len(self._tasks))
            for task in self._tasks.values():
                task.cancel()
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
            self._tasks.clear()
        self._executor.shutdown(wait=False)
        logger.info("StreamProcessor stopped")

    # ── Registry watcher ──────────────────────────────────────────

    async def _watch_registry(self) -> None:
        while self._running:
            manager = app_state.camera_manager
            if manager is None:
                logger.debug("camera manager not yet initialised — waiting")
                await asyncio.sleep(SYNC_INTERVAL)
                continue

            registered = {
                cam["camera_id"]: cam
                for cam in manager.get_status()
                if cam.get("rtsp_url")
            }

            for cid, cam in registered.items():
                existing = self._tasks.get(cid)
                if existing is None or existing.done():
                    if existing and existing.done() and not existing.cancelled():
                        exc = existing.exception()
                        if exc:
                            logger.warning("camera %s task exited with error: %s — restarting", cid, exc)
                    self._analyzers.setdefault(cid, CrowdDensityAnalyzer())
                    self._tasks[cid] = asyncio.create_task(
                        self._camera_loop(cam), name=f"cam-{cid}"
                    )
                    logger.info("StreamProcessor: launched task  cam=%s  url=%s",
                                cid, cam["rtsp_url"])

            for cid in list(self._tasks.keys()):
                if cid not in registered:
                    logger.info("StreamProcessor: camera removed — cancelling task  cam=%s", cid)
                    self._tasks.pop(cid).cancel()
                    self._analyzers.pop(cid, None)

            active = sum(1 for t in self._tasks.values() if not t.done())
            logger.debug("StreamProcessor: %d/%d tasks active", active, len(self._tasks))

            await asyncio.sleep(SYNC_INTERVAL)

    # ── Per-camera retry wrapper ───────────────────────────────────

    async def _camera_loop(self, cam: dict) -> None:
        """
        Outer loop for one camera. On any connection or stream error, waits
        RECONNECT_INTERVAL seconds then retries. Only exits on task cancellation.
        """
        camera_id  = cam["camera_id"]
        rtsp_url   = cam["rtsp_url"]
        zone_id    = cam["zone_id"]
        venue_id   = cam["venue_id"]

        z = store.get_zone(venue_id, zone_id)
        zone_label = z.name if z else zone_id

        while True:
            try:
                cap = await self._open_capture(rtsp_url)
                logger.info("camera connected  cam=%s  zone=%s  url=%s",
                            camera_id, zone_label, rtsp_url)
                await self._read_loop(cap, camera_id, zone_id, zone_label, venue_id)

            except asyncio.CancelledError:
                logger.info("camera task cancelled  cam=%s", camera_id)
                return
            except ConnectionError as exc:
                logger.warning("camera connection failed  cam=%s: %s", camera_id, exc)
            except cv2.error as exc:
                logger.error("camera OpenCV error  cam=%s: %s", camera_id, exc)
            except OSError as exc:
                logger.error("camera I/O error  cam=%s: %s", camera_id, exc)
            except Exception as exc:
                logger.error("camera unexpected error  cam=%s: %s", camera_id, exc, exc_info=True)

            logger.info("camera reconnecting in %ds  cam=%s", RECONNECT_INTERVAL, camera_id)
            await asyncio.sleep(RECONNECT_INTERVAL)

    # ── Per-camera read loop ───────────────────────────────────────

    async def _read_loop(
        self,
        cap: cv2.VideoCapture,
        camera_id: str,
        zone_id: str,
        zone_label: str,
        venue_id: str,
    ) -> None:
        """Inner frame loop. Raises on stream errors to trigger reconnect."""
        analyzer = self._analyzers[camera_id]
        try:
            while True:
                ret, frame = await self._read_frame(cap)
                if not ret or frame is None:
                    raise OSError(f"cap.read() returned False — stream ended  cam={camera_id}")

                result = await self._analyze(analyzer, frame)
                ts     = datetime.now(timezone.utc)
                conf   = result["avg_confidence"]

                # ── Confidence gate ────────────────────────────────
                if conf < MIN_CONFIDENCE:
                    logger.warning(
                        "low confidence — skipping  cam=%s  zone=%s  conf=%.2f  "
                        "(threshold=%.2f)",
                        camera_id, zone_label, conf, MIN_CONFIDENCE,
                    )
                    await asyncio.sleep(FRAME_INTERVAL)
                    continue

                person_count = result["person_count"]

                # ── Log every accepted reading ─────────────────────
                logger.info(
                    "cam=%-24s  zone=%-22s  people=%3d  "
                    "conf=%.2f  density=%5.1f%%  coverage=%.1f%%",
                    camera_id, zone_label, person_count,
                    conf, result["density_score"], result["zone_coverage_pct"],
                )

                # ── POST to ingest ─────────────────────────────────
                await self._post_ingest(
                    venue_id=venue_id,
                    zone_id=zone_id,
                    camera_id=camera_id,
                    person_count=person_count,
                    ts=ts,
                )

                # ── CSV audit log ──────────────────────────────────
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    self._executor,
                    _append_csv,
                    {
                        "timestamp":     ts.isoformat(),
                        "venue_id":      venue_id,
                        "zone_id":       zone_id,
                        "zone_name":     zone_label,
                        "camera_id":     camera_id,
                        "person_count":  person_count,
                        "avg_confidence":round(conf, 3),
                        "density_score": round(result["density_score"], 1),
                    },
                )

                await asyncio.sleep(FRAME_INTERVAL)

        finally:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, cap.release)

    # ── Blocking helpers ───────────────────────────────────────────

    async def _open_capture(self, rtsp_url: str) -> cv2.VideoCapture:
        loop = asyncio.get_running_loop()

        def _open() -> cv2.VideoCapture:
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap

        cap     = await loop.run_in_executor(self._executor, _open)
        is_open = await loop.run_in_executor(self._executor, cap.isOpened)
        if not is_open:
            await loop.run_in_executor(self._executor, cap.release)
            raise ConnectionError(f"cv2.VideoCapture could not open: {rtsp_url!r}")
        return cap

    async def _read_frame(self, cap: cv2.VideoCapture) -> tuple[bool, Optional[np.ndarray]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, cap.read)

    async def _analyze(self, analyzer: CrowdDensityAnalyzer, frame: np.ndarray) -> dict:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, analyzer.analyze_frame, frame)

    # ── Ingest POST ────────────────────────────────────────────────

    async def _post_ingest(
        self,
        *,
        venue_id: str,
        zone_id: str,
        camera_id: str,
        person_count: int,
        ts: datetime,
    ) -> None:
        payload = json.dumps({
            "venue_id":     venue_id,
            "zone_id":      zone_id,
            "person_count": person_count,
            "timestamp":    ts.isoformat(),
            "camera_id":    camera_id,
        }).encode()

        url = f"{self._ingest_base}/ingest/camera"
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        loop = asyncio.get_running_loop()
        try:
            def _do_post():
                with urllib.request.urlopen(req, timeout=INGEST_TIMEOUT) as resp:
                    return resp.status
            status = await loop.run_in_executor(self._executor, _do_post)
            logger.debug("ingest POST → HTTP %d  cam=%s", status, camera_id)
        except urllib.error.HTTPError as exc:
            logger.warning("ingest POST failed  cam=%s: HTTP %d %s",
                           camera_id, exc.code, exc.reason)
        except Exception as exc:
            logger.warning("ingest POST failed  cam=%s: %s", camera_id, exc)
