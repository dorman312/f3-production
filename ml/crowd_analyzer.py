import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CrowdDensityAnalyzer:
    MIN_CONTOUR_AREA = 800
    MAX_PERSONS = 500
    DENSITY_MAX_COVERAGE_PCT = 40.0

    def __init__(self) -> None:
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=50,
            varThreshold=40,
            detectShadows=False,
        )
        self._kernel = np.ones((5, 5), np.uint8)
        self._frame_count = 0

    def analyze_frame(self, frame: np.ndarray) -> dict:
        if frame is None or frame.size == 0:
            return {"person_count": 0, "density_score": 0.0, "zone_coverage_pct": 0.0}

        h, w = frame.shape[:2]
        total_pixels = h * w

        fg_mask = self._bg.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._kernel)
        fg_mask = cv2.dilate(fg_mask, self._kernel, iterations=1)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        persons = [c for c in contours if cv2.contourArea(c) >= self.MIN_CONTOUR_AREA]
        person_count = min(len(persons), self.MAX_PERSONS)

        fg_pixels = int(np.count_nonzero(fg_mask))
        zone_coverage_pct = round((fg_pixels / total_pixels) * 100.0, 1)

        density_from_coverage = min(
            100.0, (zone_coverage_pct / self.DENSITY_MAX_COVERAGE_PCT) * 100.0
        )
        density_from_count = min(100.0, (person_count / 100.0) * 100.0)
        density_score = round((density_from_coverage + density_from_count) / 2.0, 1)

        self._frame_count += 1
        logger.debug(
            "frame %d: persons=%d coverage=%.1f%% density=%.1f",
            self._frame_count,
            person_count,
            zone_coverage_pct,
            density_score,
        )

        return {
            "person_count": person_count,
            "density_score": density_score,
            "zone_coverage_pct": zone_coverage_pct,
        }
