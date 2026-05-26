import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:  # pragma: no cover
    _YOLO_AVAILABLE = False
    logger.warning("ultralytics not installed — YOLO detection unavailable")


class CrowdDensityAnalyzer:
    """
    YOLOv8-based crowd density analyzer.

    One instance per camera.  Thread-safety: the underlying YOLO model is not
    thread-safe; callers must serialize calls to analyze_frame / queue_analyzer
    per instance (the camera service already does this via a per-camera dict).
    """

    PERSON_CLASS = 0          # COCO class index for "person"
    CONF_THRESHOLD = 0.5      # minimum detection confidence
    MAX_PERSONS = 200         # upper bound used for density_score normalization
    DEFAULT_THROUGHPUT = 3.0  # persons per minute (service-level default)
    _EMA_ALPHA = 0.3          # weight given to newest calibration observation

    # --------------------------------------------------------------------- #
    # Construction                                                            #
    # --------------------------------------------------------------------- #

    def __init__(self, model_path: str = "yolov8n.pt") -> None:
        self._model: Optional[object] = None   # YOLO instance or None
        self._frame_count: int = 0
        # zone_id -> calibrated throughput (persons / minute)
        self._throughput_rates: dict[str, float] = {}

        if _YOLO_AVAILABLE:
            try:
                self._model = YOLO(model_path)
                logger.info("YOLOv8 model loaded: %s", model_path)
            except Exception as exc:
                logger.error("Failed to load YOLO model %s: %s", model_path, exc)
        else:
            logger.warning(
                "Running without YOLO — install ultralytics to enable person detection"
            )

    # --------------------------------------------------------------------- #
    # Public API                                                              #
    # --------------------------------------------------------------------- #

    def analyze_frame(self, frame: np.ndarray) -> dict:
        """
        Run YOLOv8 person detection on a full frame.

        Returns
        -------
        person_count       : int   — confirmed person detections
        density_score      : float — 0-100 scale relative to MAX_PERSONS
        bounding_box_count : int   — same as person_count (explicit for callers)
        avg_confidence     : float — mean detection confidence (0-1)
        zone_coverage_pct  : float — fraction of frame area covered by person boxes
        """
        empty = {
            "person_count": 0,
            "density_score": 0.0,
            "bounding_box_count": 0,
            "avg_confidence": 0.0,
            "zone_coverage_pct": 0.0,
        }
        if frame is None or frame.size == 0:
            return empty
        if self._model is None:
            return empty

        h, w = frame.shape[:2]
        frame_area = float(h * w)

        persons, total_box_area = self._detect_persons(frame)
        person_count = len(persons)

        avg_confidence = (
            round(sum(p["confidence"] for p in persons) / person_count, 3)
            if persons else 0.0
        )
        density_score = round(
            min(100.0, (person_count / self.MAX_PERSONS) * 100.0), 1
        )
        zone_coverage_pct = round(
            min(100.0, (total_box_area / frame_area) * 100.0), 1
        ) if frame_area > 0 else 0.0

        self._frame_count += 1
        logger.debug(
            "frame %d: persons=%d density=%.1f avg_conf=%.3f coverage=%.1f%%",
            self._frame_count, person_count, density_score,
            avg_confidence, zone_coverage_pct,
        )

        return {
            "person_count": person_count,
            "density_score": density_score,
            "bounding_box_count": person_count,
            "avg_confidence": avg_confidence,
            "zone_coverage_pct": zone_coverage_pct,
        }

    def queue_analyzer(
        self,
        frame: np.ndarray,
        queue_zone_coords: tuple[int, int, int, int],
        zone_id: str = "",
        throughput_rate: Optional[float] = None,
    ) -> dict:
        """
        Crop the frame to a queue zone and estimate wait time.

        Parameters
        ----------
        frame              : BGR frame from OpenCV
        queue_zone_coords  : (x1, y1, x2, y2) pixel coordinates of the queue region
        zone_id            : used to look up the calibrated throughput rate
        throughput_rate    : override persons/minute; if None, uses calibrated or default

        Returns
        -------
        estimated_wait_minutes : float
        queue_length           : int
        confidence_score       : float — avg detection confidence in the crop
        """
        if frame is None or frame.size == 0:
            return {"estimated_wait_minutes": 0.0, "queue_length": 0, "confidence_score": 0.0}

        x1, y1, x2, y2 = queue_zone_coords
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(fw, int(x2)), min(fh, int(y2))

        if x2 - x1 < 32 or y2 - y1 < 32:  # skip crops too small for reliable detection
            logger.warning(
                "queue_analyzer: crop too small (%dx%d) for zone=%s",
                x2 - x1, y2 - y1, zone_id,
            )
            return {"estimated_wait_minutes": 0.0, "queue_length": 0, "confidence_score": 0.0}

        crop = frame[y1:y2, x1:x2]
        analysis = self.analyze_frame(crop)

        queue_length = analysis["person_count"]
        confidence   = analysis["avg_confidence"]

        # Priority: explicit arg > per-zone calibrated rate > global default
        rate = throughput_rate
        if rate is None:
            rate = self._throughput_rates.get(zone_id, self.DEFAULT_THROUGHPUT)
        rate = max(rate, 0.1)  # guard against division by zero

        estimated_wait = round(queue_length / rate, 2)

        logger.debug(
            "queue zone=%s count=%d rate=%.2f/min est_wait=%.2fmin",
            zone_id, queue_length, rate, estimated_wait,
        )

        return {
            "estimated_wait_minutes": estimated_wait,
            "queue_length": queue_length,
            "confidence_score": confidence,
        }

    def calibrate_zone(
        self,
        zone_id: str,
        actual_wait_minutes: float,
        observed_count: int,
    ) -> float:
        """
        Update the throughput rate for a zone from a real-world observation.

        Blends the observed rate with the existing estimate using an exponential
        moving average so that predictions improve incrementally without
        over-reacting to individual noisy measurements.

        Parameters
        ----------
        zone_id             : identifier for the zone being calibrated
        actual_wait_minutes : ground-truth wait time measured at the zone
        observed_count      : number of people observed in that period

        Returns
        -------
        updated throughput rate (persons / minute) for the zone
        """
        if actual_wait_minutes <= 0 or observed_count <= 0:
            logger.warning(
                "calibrate_zone: skipping invalid inputs zone=%s "
                "wait=%.2f count=%d",
                zone_id, actual_wait_minutes, observed_count,
            )
            return self._throughput_rates.get(zone_id, self.DEFAULT_THROUGHPUT)

        observed_rate = observed_count / actual_wait_minutes
        current_rate  = self._throughput_rates.get(zone_id, self.DEFAULT_THROUGHPUT)

        updated_rate = (
            self._EMA_ALPHA * observed_rate
            + (1.0 - self._EMA_ALPHA) * current_rate
        )
        self._throughput_rates[zone_id] = updated_rate

        logger.info(
            "calibrated zone=%s: observed=%.2f/min prev=%.2f/min -> %.2f/min",
            zone_id, observed_rate, current_rate, updated_rate,
        )
        return updated_rate

    def get_zone_throughput(self, zone_id: str) -> float:
        """Return the current calibrated throughput for a zone, or the default."""
        return self._throughput_rates.get(zone_id, self.DEFAULT_THROUGHPUT)

    # --------------------------------------------------------------------- #
    # Internal helpers                                                        #
    # --------------------------------------------------------------------- #

    def _detect_persons(
        self, frame: np.ndarray
    ) -> tuple[list[dict], float]:
        """
        Run YOLO inference and return (person_detections, total_box_area).

        Each detection dict contains {"confidence": float, "bbox": (x1,y1,x2,y2)}.
        """
        if self._model is None:
            return [], 0.0

        results = self._model(frame, verbose=False)
        persons: list[dict] = []
        total_box_area = 0.0

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                if cls != self.PERSON_CLASS or conf < self.CONF_THRESHOLD:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                persons.append({"confidence": conf, "bbox": (x1, y1, x2, y2)})
                total_box_area += (x2 - x1) * (y2 - y1)

        return persons, total_box_area
