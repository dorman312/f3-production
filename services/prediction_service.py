import logging
from datetime import datetime, timezone

from state import store

logger = logging.getLogger(__name__)

_ZONE_PEAK_HOURS: dict[str, frozenset] = {
    "gate":       frozenset({7, 8, 17, 18, 19}),
    "gates":      frozenset({7, 8, 17, 18, 19}),
    "concession": frozenset({12, 13, 18, 19, 20}),
    "concessions":frozenset({12, 13, 18, 19, 20}),
    "restroom":   frozenset({12, 13, 19, 20, 21}),
    "restrooms":  frozenset({12, 13, 19, 20, 21}),
    "bathroom":   frozenset({12, 13, 19, 20, 21}),
    "bathrooms":  frozenset({12, 13, 19, 20, 21}),
    "exit":       frozenset({17, 18, 22, 23}),
    "exits":      frozenset({17, 18, 22, 23}),
    "corridor":   frozenset({8, 9, 12, 13, 17, 18}),
}


class VenuePredictionService:
    def _get_flow_direction(self, venue_id: str, zone_id: str) -> str:
        zone_readings = [
            r for r in store.readings
            if r.venue_id == venue_id and r.zone_id == zone_id
        ]
        if len(zone_readings) < 2:
            return "stable"
        zone_readings.sort(key=lambda r: r.timestamp)
        recent = zone_readings[-3:]
        delta = recent[-1].person_count - recent[0].person_count
        if delta > 5:
            return "increasing"
        if delta < -5:
            return "decreasing"
        return "stable"

    def _density_rate(
        self,
        zone_type: str,
        current_hour: int,
        current_density: float,
        flow_dir: str,
    ) -> float:
        peak_hours = _ZONE_PEAK_HOURS.get(zone_type.lower(), frozenset())
        in_peak = current_hour in peak_hours
        approaching = any((current_hour + 1) % 24 == h for h in peak_hours)
        leaving = any((current_hour - 1) % 24 == h for h in peak_hours)

        if flow_dir == "increasing":
            base = 0.60
        elif flow_dir == "decreasing":
            base = -0.50
        else:
            base = 0.05

        if in_peak and current_density < 70:
            base = max(base, 0.40)
        elif leaving:
            base = min(base, -0.20)
        elif approaching:
            base = max(base, 0.20)

        mean_pull = (50.0 - current_density) * 0.008
        return base + mean_pull

    def predict_next_30min(self, venue_id: str) -> dict | None:
        if not store.get_venue(venue_id):
            return None

        zone_map = store.get_zones(venue_id)
        now = datetime.now(timezone.utc)
        hour = now.hour
        forecasts: dict[str, dict] = {}

        for zone in zone_map.values():
            current_density, _ = store.calculate_density(venue_id, zone.id)
            flow_dir = self._get_flow_direction(venue_id, zone.id)
            rate = self._density_rate(zone.type, hour, current_density, flow_dir)

            intervals: dict[str, float] = {}
            for minutes_ahead in (5, 10, 15, 20, 25, 30):
                predicted = current_density + rate * minutes_ahead
                intervals[f"{minutes_ahead}min"] = round(
                    max(0.0, min(100.0, predicted)), 1
                )

            forecasts[zone.id] = {
                "zone_name": zone.name,
                "zone_type": zone.type,
                "current_density": current_density,
                "flow_direction": flow_dir,
                "forecast": intervals,
            }

        return {
            "venue_id": venue_id,
            "generated_at": now.isoformat(),
            "forecasts": forecasts,
        }

    def peak_periods(self, venue_id: str) -> dict | None:
        venue = store.get_venue(venue_id)
        if not venue:
            return None

        zone_map = store.get_zones(venue_id)
        zone_types = {z.type.lower() for z in zone_map.values()}
        now = datetime.now(timezone.utc)

        _concession = {"concession", "concessions", "food", "cafe"}
        _gate = {"gate", "gates", "entrance", "check-in"}
        _restroom = {"restroom", "restrooms", "bathroom", "bathrooms"}
        _exit = {"exit", "exits"}

        peaks: list[dict] = []

        if zone_types & _concession:
            matched = list(zone_types & _concession)
            peaks.append({"time": "12:00-14:00", "label": "Lunch rush",
                          "intensity": "high", "affected_types": matched})
            peaks.append({"time": "18:00-20:00", "label": "Dinner rush",
                          "intensity": "high", "affected_types": matched})

        if zone_types & _gate:
            matched = list(zone_types & _gate)
            peaks.append({"time": "07:00-09:00", "label": "Morning arrivals",
                          "intensity": "medium", "affected_types": matched})
            peaks.append({"time": "17:00-19:00", "label": "Evening departures",
                          "intensity": "high",
                          "affected_types": list((zone_types & _gate) | (zone_types & _exit))})

        if zone_types & _restroom:
            matched = list(zone_types & _restroom)
            peaks.append({"time": "12:30-13:30", "label": "Post-lunch restroom rush",
                          "intensity": "medium", "affected_types": matched})

        if zone_types & _exit:
            matched = list(zone_types & _exit)
            peaks.append({"time": "17:00-19:00", "label": "End-of-day exodus",
                          "intensity": "high", "affected_types": matched})

        peaks.sort(key=lambda p: p["time"])

        seen: set[tuple] = set()
        unique: list[dict] = []
        for p in peaks:
            key = (p["time"], p["label"])
            if key not in seen:
                seen.add(key)
                unique.append(p)

        current_hour = now.hour
        current_peak: dict | None = None
        for peak in unique:
            start_str, end_str = peak["time"].split("-")
            start_h = int(start_str.split(":")[0])
            end_h = int(end_str.split(":")[0])
            if start_h <= current_hour < end_h:
                current_peak = peak
                break

        return {
            "venue_id": venue_id,
            "venue_type": venue.type,
            "peaks_today": unique,
            "current_peak": current_peak,
            "generated_at": now.isoformat(),
        }

    def crowd_flow_direction(self, venue_id: str, zone_id: str) -> dict | None:
        zone = store.get_zone(venue_id, zone_id)
        if not zone:
            return None

        zone_readings = [
            r for r in store.readings
            if r.venue_id == venue_id and r.zone_id == zone_id
        ]

        _, current_persons = store.calculate_density(venue_id, zone_id)

        if len(zone_readings) < 2:
            return {
                "direction": "stable",
                "rate_per_minute": 0.0,
                "confidence": "low",
                "readings_analyzed": len(zone_readings),
                "current_persons": current_persons,
            }

        zone_readings.sort(key=lambda r: r.timestamp)
        recent = zone_readings[-min(5, len(zone_readings)):]
        delta = recent[-1].person_count - recent[0].person_count

        try:
            dt_seconds = (
                recent[-1].timestamp - recent[0].timestamp
            ).total_seconds()
            rate = delta / (dt_seconds / 60.0) if dt_seconds > 0 else 0.0
        except TypeError:
            rate = 0.0

        if delta > 5:
            direction = "increasing"
        elif delta < -5:
            direction = "decreasing"
        else:
            direction = "stable"

        n = len(zone_readings)
        confidence = "high" if n >= 5 else ("medium" if n >= 3 else "low")

        return {
            "direction": direction,
            "rate_per_minute": round(rate, 2),
            "confidence": confidence,
            "readings_analyzed": n,
            "current_persons": current_persons,
        }
