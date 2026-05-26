"""
workers/stream_processor.py

Background asyncio worker that polls every registered RTSP camera
once per FRAME_INTERVAL seconds, runs YOLOv8 person detection on
each frame, and posts the result to POST /ingest/camera.

Integration with the FastAPI app (add to main.py lifespan):

    from workers.stream_processor import StreamProcessor
    processor = StreamProcessor()
    asyncio.create_task(processor.start())

The worker calls the running server's own HTTP endpoint so that the
full ingest pipeline (store write, alert generation) fires identically
to any other data source.
"""

import asyncio
import json
import logging
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np

from ml.crowd_analyzer import CrowdDensityAnalyzer
from state import app_state
from state import store

logger = logging.getLogger(__name__)

# ── Tuning constants ──────────────────────────────────────────────
FRAME_INTERVAL  = 30     # seconds between frame captures per camera
SYNC_INTERVAL   = 30     # seconds between camera-registry sync passes
INITIAL_BACKOFF = 5      # seconds before first reconnect attempt
MAX_BACKOFF     = 300    # cap reconnect wait at 5 minutes
INGEST_TIMEOUT  = 5      # seconds for the HTTP POST to /ingest/camera


class StreamProcessor:
    """
    Manages one asyncio task per registered camera.

    On each tick:
      1. Grab a single frame from the RTSP stream (blocking → thread pool).
      2. Run CrowdDensityAnalyzer.analyze_frame() (CPU-bound → thread pool).
      3. POST the result to POST /ingest/camera.

    A registry-watcher coroutine re-runs every SYNC_INTERVAL seconds so that
    cameras registered or removed at runtime are picked up automatically.
    """

    def __init__(self, ingest_base: str = "http://localhost:8000") -> None:
        self._ingest_base = ingest_base.rstrip("/")
        # One task per camera_id
        self._tasks:     dict[str, asyncio.Task]          = {}
        # One analyzer per camera_id (YOLO models are not thread-safe; each
        # camera owns its instance and calls are serialised within that task)
        self._analyzers: dict[str, CrowdDensityAnalyzer]  = {}
        # Shared thread pool for all blocking I/O and CPU work
        self._executor   = ThreadPoolExecutor(
            max_workers=16, thread_name_prefix="rtsp-worker"
        )
        self._running = False

    # ── Lifecycle ─────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the registry watcher. Blocks until stop() is called."""
        self._running = True
        logger.info("StreamProcessor starting — ingest endpoint: %s", self._ingest_base)
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
        """
        Runs continuously. Each pass:
          - Starts a new camera task for any camera that appeared in the registry.
          - Cancels the task for any camera that was removed.
          - Restarts tasks that exited unexpectedly (e.g. after exhausting retries).
        """
        while self._running:
            manager = app_state.camera_manager
            if manager is None:
                logger.debug("Camera manager not yet initialised — waiting")
                await asyncio.sleep(SYNC_INTERVAL)
                continue

            registered = {
                cam["camera_id"]: cam
                for cam in manager.get_status()
                if cam.get("rtsp_url")
            }

            # Start or restart tasks
            for cid, cam in registered.items():
                existing = self._tasks.get(cid)
                if existing is None or existing.done():
                    if existing and existing.done() and not existing.cancelled():
                        exc = existing.exception()
                        if exc:
                            logger.warning("camera %s task exited with error: %s — restarting", cid, exc)
                    self._analyzers.setdefault(cid, CrowdDensityAnalyzer())
                    task = asyncio.create_task(
                        self._camera_loop(cam), name=f"cam-{cid}"
                    )
                    self._tasks[cid] = task
                    logger.info("StreamProcessor: launched task for camera %s → %s", cid, cam["rtsp_url"])

            # Cancel tasks for removed cameras
            for cid in list(self._tasks.keys()):
                if cid not in registered:
                    logger.info("StreamProcessor: camera %s removed — cancelling task", cid)
                    self._tasks.pop(cid).cancel()
                    self._analyzers.pop(cid, None)

            active = sum(1 for t in self._tasks.values() if not t.done())
            logger.debug("StreamProcessor: %d/%d camera tasks active", active, len(self._tasks))

            await asyncio.sleep(SYNC_INTERVAL)

    # ── Per-camera retry wrapper ───────────────────────────────────

    async def _camera_loop(self, cam: dict) -> None:
        """
        Outer loop for one camera. Reconnects with exponential back-off on
        any connection or read failure. Only exits when the task is cancelled.
        """
        camera_id  = cam["camera_id"]
        rtsp_url   = cam["rtsp_url"]
        zone_id    = cam["zone_id"]
        venue_id   = cam["venue_id"]

        # Resolve a human-readable zone name for logs
        z = store.get_zone(venue_id, zone_id)
        zone_label = z.name if z else zone_id

        backoff = INITIAL_BACKOFF

        while True:
            try:
                cap = await self._open_capture(rtsp_url)
                backoff = INITIAL_BACKOFF   # reset after successful connect
                logger.info(
                    "camera %s connected — reading every %ds  zone=%s",
                    camera_id, FRAME_INTERVAL, zone_label,
                )
                await self._read_loop(cap, camera_id, zone_id, zone_label, venue_id)

            except asyncio.CancelledError:
                logger.info("camera %s task cancelled", camera_id)
                return
            except ConnectionError as exc:
                logger.warning("camera %s: %s", camera_id, exc)
            except cv2.error as exc:
                logger.error("camera %s OpenCV error: %s", camera_id, exc)
            except OSError as exc:
                logger.error("camera %s I/O error: %s", camera_id, exc)
            except Exception as exc:
                logger.error("camera %s unexpected error: %s", camera_id, exc, exc_info=True)

            logger.info("camera %s — reconnecting in %ds", camera_id, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)

    # ── Per-camera read loop ───────────────────────────────────────

    async def _read_loop(
        self,
        cap: cv2.VideoCapture,
        camera_id: str,
        zone_id: str,
        zone_label: str,
        venue_id: str,
    ) -> None:
        """
        Inner frame loop. Raises on stream errors so _camera_loop triggers
        a reconnect. Always releases cap on exit.
        """
        analyzer = self._analyzers[camera_id]
        try:
            while True:
                ret, frame = await self._read_frame(cap)
                if not ret or frame is None:
                    raise OSError(f"camera {camera_id}: cap.read() returned False — stream ended")

                result = await self._analyze(analyzer, frame)

                await self._post_ingest(
                    venue_id=venue_id,
                    zone_id=zone_id,
                    camera_id=camera_id,
                    person_count=result["person_count"],
                )

                logger.info(
                    "cam=%-24s  zone=%-22s  people=%3d  "
                    "density=%5.1f%%  conf=%.2f  coverage=%.1f%%",
                    camera_id, zone_label,
                    result["person_count"],
                    result["density_score"],
                    result["avg_confidence"],
                    result["zone_coverage_pct"],
                )

                await asyncio.sleep(FRAME_INTERVAL)

        finally:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, cap.release)

    # ── Blocking helpers (all run in thread pool) ──────────────────

    async def _open_capture(self, rtsp_url: str) -> cv2.VideoCapture:
        """Open an RTSP stream. Raises ConnectionError if it cannot be opened."""
        loop = asyncio.get_running_loop()

        def _open() -> cv2.VideoCapture:
            cap = cv2.VideoCapture(rtsp_url)
            # For RTSP, prefer TCP over UDP to reduce packet loss
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap

        cap = await loop.run_in_executor(self._executor, _open)

        is_open = await loop.run_in_executor(self._executor, cap.isOpened)
        if not is_open:
            await loop.run_in_executor(self._executor, cap.release)
            raise ConnectionError(f"cv2.VideoCapture could not open: {rtsp_url!r}")

        return cap

    async def _read_frame(
        self, cap: cv2.VideoCapture
    ) -> tuple[bool, Optional[np.ndarray]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, cap.read)

    async def _analyze(
        self, analyzer: CrowdDensityAnalyzer, frame: np.ndarray
    ) -> dict:
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
    ) -> None:
        """POST a CameraIngest payload to /ingest/camera."""
        payload = json.dumps({
            "venue_id":     venue_id,
            "zone_id":      zone_id,
            "person_count": person_count,
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "camera_id":    camera_id,
        }).encode()

        url = f"{self._ingest_base}/ingest/camera"
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        loop = asyncio.get_running_loop()
        try:
            def _do_post():
                with urllib.request.urlopen(req, timeout=INGEST_TIMEOUT) as resp:
                    return resp.status

            status = await loop.run_in_executor(self._executor, _do_post)
            logger.debug("ingest POST cam=%s → HTTP %d", camera_id, status)
        except urllib.error.HTTPError as exc:
            logger.warning(
                "ingest POST failed cam=%s: HTTP %d %s", camera_id, exc.code, exc.reason
            )
        except Exception as exc:
            logger.warning("ingest POST failed cam=%s: %s", camera_id, exc)
