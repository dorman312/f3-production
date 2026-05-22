import logging
from datetime import datetime, timezone

from state import app_state
from state import store

logger = logging.getLogger(__name__)

_RISK_COLORS: dict[str, str] = {
    "low": "#22c55e",
    "medium": "#f59e0b",
    "high": "#ef4444",
}


class VenueMapService:
    def generate_map(self, venue_id: str) -> dict | None:
        venue = store.get_venue(venue_id)
        if not venue:
            return None

        zone_map = store.get_zones(venue_id)
        active_alerts = [
            a for a in store.alerts.get(venue_id, {}).values() if not a.resolved
        ]
        hour = datetime.now(timezone.utc).hour

        zones_out: list[dict] = []
        density_pairs: list[tuple[str, float]] = []
        wait_sum = 0.0

        for zone in zone_map.values():
            density_score, person_count = store.calculate_density(venue_id, zone.id)
            risk_level = store.get_risk_level(density_score)

            if app_state.predictor is not None:
                pred = app_state.predictor.predict_with_confidence(
                    zone.type, density_score, hour
                )
                wait_minutes = pred["predicted"]
            else:
                wait_minutes = round(
                    density_score * store.get_wait_factor(zone.type), 1
                )

            density_pairs.append((zone.id, density_score))
            wait_sum += wait_minutes

            zones_out.append(
                {
                    "id": zone.id,
                    "name": zone.name,
                    "type": zone.type,
                    "x": zone.x_pct,
                    "y": zone.y_pct,
                    "width": zone.w_pct,
                    "height": zone.h_pct,
                    "current_density": density_score,
                    "density_score": density_score,
                    "risk_level": risk_level,
                    "wait_minutes": round(wait_minutes, 1),
                    "person_count": person_count,
                    "color_hex": _RISK_COLORS[risk_level],
                    "is_accessible": True,
                    "adjacent_zone_ids": zone.adjacent_zone_ids,
                }
            )

        total_cap = sum(z.max_capacity for z in zone_map.values())
        high_risk = sum(1 for _, ds in density_pairs if ds >= 90)
        avg_wait = round(wait_sum / len(density_pairs), 1) if density_pairs else 0.0
        busiest = max(density_pairs, key=lambda t: t[1], default=(None, 0))
        safest = min(density_pairs, key=lambda t: t[1], default=(None, 0))

        return {
            "venue": {
                "id": venue.id,
                "name": venue.name,
                "type": venue.type,
                "capacity": total_cap,
            },
            "zones": zones_out,
            "alerts": [
                {
                    "zone_id": a.zone_id,
                    "severity": a.level,
                    "message": a.message,
                    "created_at": a.created_at.isoformat(),
                }
                for a in active_alerts
            ],
            "summary": {
                "total_zones": len(zone_map),
                "high_risk_zones": high_risk,
                "avg_wait_minutes": avg_wait,
                "busiest_zone_id": busiest[0],
                "safest_zone_id": safest[0],
            },
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    def get_zone_detail(self, venue_id: str, zone_id: str) -> dict | None:
        if not store.get_venue(venue_id):
            return None
        zone = store.get_zone(venue_id, zone_id)
        if not zone:
            return None

        density_score, person_count = store.calculate_density(venue_id, zone_id)
        risk_level = store.get_risk_level(density_score)
        hour = datetime.now(timezone.utc).hour

        if app_state.predictor is not None:
            pred = app_state.predictor.predict_with_confidence(
                zone.type, density_score, hour
            )
            wait_minutes = pred["predicted"]
            confidence_interval = {
                "low": pred["confidence_low"],
                "high": pred["confidence_high"],
                "model_trained": pred["model_trained"],
            }
        else:
            wait_minutes = round(
                density_score * store.get_wait_factor(zone.type), 1
            )
            confidence_interval = None

        return {
            "id": zone.id,
            "name": zone.name,
            "type": zone.type,
            "venue_id": venue_id,
            "x": zone.x_pct,
            "y": zone.y_pct,
            "width": zone.w_pct,
            "height": zone.h_pct,
            "area_m2": zone.area_m2,
            "max_capacity": zone.max_capacity,
            "current_density": density_score,
            "density_score": density_score,
            "risk_level": risk_level,
            "wait_minutes": round(wait_minutes, 1),
            "person_count": person_count,
            "color_hex": _RISK_COLORS[risk_level],
            "is_accessible": True,
            "adjacent_zone_ids": zone.adjacent_zone_ids,
            "confidence_interval": confidence_interval,
        }
