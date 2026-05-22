import logging
import math

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

logger = logging.getLogger(__name__)

_ZONE_TYPES = ("gate", "concession", "restroom", "exit", "corridor")
_ZONE_IDX: dict[str, int] = {z: i for i, z in enumerate(_ZONE_TYPES)}

_ZONE_PARAMS: dict[str, dict] = {
    "gate": {
        "base_wait": 2.0,
        "density_factor": 0.08,
        "peak_hours": frozenset({7, 8, 17, 18, 19}),
        "peak_bonus": 2.0,
        "noise_std": 0.5,
        "hist_density": 45.0,
    },
    "concession": {
        "base_wait": 3.0,
        "density_factor": 0.15,
        "peak_hours": frozenset({12, 13, 18, 19, 20}),
        "peak_bonus": 4.0,
        "noise_std": 1.0,
        "hist_density": 60.0,
    },
    "restroom": {
        "base_wait": 1.5,
        "density_factor": 0.12,
        "peak_hours": frozenset({12, 13, 19, 20, 21}),
        "peak_bonus": 2.5,
        "noise_std": 0.7,
        "hist_density": 50.0,
    },
    "exit": {
        "base_wait": 1.0,
        "density_factor": 0.10,
        "peak_hours": frozenset({17, 18, 22, 23}),
        "peak_bonus": 3.0,
        "noise_std": 0.4,
        "hist_density": 35.0,
    },
    "corridor": {
        "base_wait": 0.5,
        "density_factor": 0.05,
        "peak_hours": frozenset({8, 9, 12, 13, 17, 18}),
        "peak_bonus": 0.8,
        "noise_std": 0.3,
        "hist_density": 40.0,
    },
}

_FALLBACK_FACTORS: dict[str, float] = {
    "gate": 0.08,
    "gates": 0.08,
    "concession": 0.15,
    "concessions": 0.15,
    "restroom": 0.12,
    "restrooms": 0.12,
    "bathroom": 0.12,
    "bathrooms": 0.12,
    "exit": 0.10,
    "exits": 0.10,
    "corridor": 0.05,
}


class WaitTimePredictor:
    N_SAMPLES = 5000

    def __init__(self) -> None:
        self._model: GradientBoostingRegressor | None = None
        self._model_low: GradientBoostingRegressor | None = None
        self._model_high: GradientBoostingRegressor | None = None
        self._trained = False

    def train(self) -> None:
        logger.info("WaitTimePredictor: generating %d synthetic training samples", self.N_SAMPLES)
        X, y = self._generate_training_data()

        self._model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
        )
        self._model_low = GradientBoostingRegressor(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=3,
            loss="quantile",
            alpha=0.10,
            random_state=42,
        )
        self._model_high = GradientBoostingRegressor(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=3,
            loss="quantile",
            alpha=0.90,
            random_state=42,
        )

        self._model.fit(X, y)
        self._model_low.fit(X, y)
        self._model_high.fit(X, y)
        self._trained = True
        logger.info("WaitTimePredictor: training complete — 3 GBR models ready")

    def _generate_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed=42)
        zone_keys = list(_ZONE_PARAMS.keys())
        X_rows: list[list[float]] = []
        y_rows: list[float] = []

        for _ in range(self.N_SAMPLES):
            zone_type = str(rng.choice(zone_keys))
            p = _ZONE_PARAMS[zone_type]
            density = float(rng.uniform(0.0, 100.0))
            hour = int(rng.integers(0, 24))
            dow = int(rng.integers(0, 7))
            hist = float(np.clip(p["hist_density"] + rng.normal(0, 5), 0, 100))

            wait = p["base_wait"] + p["density_factor"] * density
            if hour in p["peak_hours"]:
                wait += p["peak_bonus"] * (density / 100.0)
            if dow >= 5:
                wait *= 1.2
            wait += float(rng.normal(0, p["noise_std"]))
            wait = max(0.0, wait)

            X_rows.append(self._featurize_row(zone_type, density, hour, dow, hist))
            y_rows.append(wait)

        return np.array(X_rows, dtype=np.float64), np.array(y_rows, dtype=np.float64)

    def _featurize_row(
        self,
        zone_type: str,
        density: float,
        hour: int,
        dow: int,
        hist_density: float,
    ) -> list[float]:
        zone_idx = float(_ZONE_IDX.get(zone_type.lower(), 0))
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        return [density, zone_idx, hour_sin, hour_cos, float(dow), hist_density]

    def _make_X(self, zone_type: str, density: float, hour: int) -> np.ndarray:
        p = _ZONE_PARAMS.get(zone_type.lower(), {})
        hist = float(p.get("hist_density", 50.0))
        row = self._featurize_row(zone_type, density, hour, 1, hist)
        return np.array([row], dtype=np.float64)

    def _fallback(self, zone_type: str, density: float) -> float:
        factor = _FALLBACK_FACTORS.get(zone_type.lower(), 0.10)
        return round(density * factor, 1)

    def predict(self, zone_type: str, density: float, hour: int) -> float:
        if not self._trained or self._model is None:
            return self._fallback(zone_type, density)
        X = self._make_X(zone_type, density, hour)
        return float(max(0.0, round(float(self._model.predict(X)[0]), 1)))

    def predict_with_confidence(
        self, zone_type: str, density: float, hour: int
    ) -> dict:
        if not self._trained or self._model is None:
            val = self._fallback(zone_type, density)
            return {
                "predicted": val,
                "confidence_low": round(val * 0.8, 1),
                "confidence_high": round(val * 1.2, 1),
                "model_trained": False,
            }
        X = self._make_X(zone_type, density, hour)
        predicted = max(0.0, float(self._model.predict(X)[0]))
        low = max(0.0, float(self._model_low.predict(X)[0]))
        high = max(predicted, float(self._model_high.predict(X)[0]))
        return {
            "predicted": round(predicted, 1),
            "confidence_low": round(low, 1),
            "confidence_high": round(high, 1),
            "model_trained": True,
        }
