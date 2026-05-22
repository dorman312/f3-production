import heapq
import itertools
import logging
from collections import deque
from datetime import datetime, timezone

from state import app_state
from state import store

logger = logging.getLogger(__name__)

_BASE_TRAVEL_TIME = 1.5
_DENSITY_PENALTY = 0.05


class SmartRoutingService:
    def _edge_weight(self, venue_id: str, zone_id: str) -> float:
        density_score, _ = store.calculate_density(venue_id, zone_id)
        return _BASE_TRAVEL_TIME + _DENSITY_PENALTY * density_score

    def _dijkstra(
        self,
        venue_id: str,
        start: str,
        target: str | None = None,
    ) -> dict[str, tuple[float, list[str]]]:
        zone_map = store.get_zones(venue_id)
        if start not in zone_map:
            return {}

        counter = itertools.count()
        heap: list[tuple[float, int, str, list[str]]] = [
            (0.0, next(counter), start, [start])
        ]
        distances: dict[str, tuple[float, list[str]]] = {}

        while heap:
            cost, _, current, path = heapq.heappop(heap)

            if current in distances:
                continue
            distances[current] = (cost, path)

            if target is not None and current == target:
                return {target: (cost, path)}

            zone = zone_map.get(current)
            if not zone:
                continue

            for neighbor_id in zone.adjacent_zone_ids:
                if neighbor_id in distances or neighbor_id not in zone_map:
                    continue
                w = self._edge_weight(venue_id, neighbor_id)
                heapq.heappush(
                    heap,
                    (cost + w, next(counter), neighbor_id, path + [neighbor_id]),
                )

        return distances

    def _bfs_avoiding_high(
        self,
        zone_map: dict,
        start: str,
        end: str,
        venue_id: str,
    ) -> list[str] | None:
        queue: deque[list[str]] = deque([[start]])
        visited: set[str] = {start}
        while queue:
            path = queue.popleft()
            current = path[-1]
            if current == end:
                return path
            zone = zone_map.get(current)
            if not zone:
                continue
            for neighbor_id in zone.adjacent_zone_ids:
                if neighbor_id in visited or neighbor_id not in zone_map:
                    continue
                if neighbor_id != end:
                    ds, _ = store.calculate_density(venue_id, neighbor_id)
                    if ds >= 90:
                        continue
                visited.add(neighbor_id)
                queue.append(path + [neighbor_id])
        return None

    def _build_directions(self, path: list[str], zone_map: dict) -> list[str]:
        dirs: list[str] = []
        for i, zid in enumerate(path):
            zone = zone_map.get(zid)
            name = zone.name if zone else zid
            if i == 0:
                dirs.append(f"Start at {name}")
            elif i == len(path) - 1:
                dirs.append(f"Arrive at {name}")
            else:
                dirs.append(f"Continue through {name}")
        return dirs

    def find_route(
        self, venue_id: str, from_zone: str, to_zone: str
    ) -> dict | None:
        zone_map = store.get_zones(venue_id)
        if from_zone not in zone_map or to_zone not in zone_map:
            return None

        if from_zone == to_zone:
            name = zone_map[from_zone].name
            return {
                "path": [from_zone],
                "zone_names": [name],
                "estimated_minutes": 0.0,
                "congestion_avoided": True,
                "alternative_path": None,
                "directions": [f"You are already at {name}"],
            }

        result = self._dijkstra(venue_id, from_zone, target=to_zone)
        if to_zone not in result:
            return None

        cost, path = result[to_zone]

        congestion_avoided = not any(
            store.get_risk_level(store.calculate_density(venue_id, zid)[0]) == "high"
            for zid in path[1:-1]
        )

        alt = self._bfs_avoiding_high(zone_map, from_zone, to_zone, venue_id)
        alt_path = alt if (alt and alt != path) else None

        return {
            "path": path,
            "zone_names": [zone_map[zid].name for zid in path if zid in zone_map],
            "estimated_minutes": round(cost, 1),
            "congestion_avoided": congestion_avoided,
            "alternative_path": alt_path,
            "directions": self._build_directions(path, zone_map),
        }

    def find_nearest(
        self, venue_id: str, from_zone: str, zone_type: str
    ) -> dict | None:
        zone_map = store.get_zones(venue_id)
        if from_zone not in zone_map:
            return None

        type_lower = zone_type.lower()
        targets = [
            zid
            for zid, z in zone_map.items()
            if z.type.lower() == type_lower
            or z.type.lower().rstrip("s") == type_lower.rstrip("s")
        ]
        if not targets:
            return None

        distances = self._dijkstra(venue_id, from_zone)
        hour = datetime.now(timezone.utc).hour
        best: dict | None = None
        best_total = float("inf")

        for target_id in targets:
            if target_id not in distances:
                continue
            walk_cost, path = distances[target_id]
            target_zone = zone_map[target_id]
            ds, _ = store.calculate_density(venue_id, target_id)

            if app_state.predictor is not None:
                wait = app_state.predictor.predict(target_zone.type, ds, hour)
            else:
                wait = round(ds * store.get_wait_factor(target_zone.type), 1)

            total = walk_cost + wait
            if total < best_total:
                best_total = total
                best = {
                    "zone_id": target_id,
                    "zone_name": target_zone.name,
                    "zone_type": target_zone.type,
                    "walk_minutes": round(walk_cost, 1),
                    "wait_minutes": round(wait, 1),
                    "total_minutes": round(total, 1),
                    "path": path,
                    "zone_names": [
                        zone_map[zid].name for zid in path if zid in zone_map
                    ],
                }

        return best
