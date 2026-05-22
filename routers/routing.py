import logging
from collections import deque
from fastapi import APIRouter, HTTPException, Query
from models.venue import RouteResponse, Zone
from state import app_state
from state import store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/venues", tags=["routing"])


def _bfs(
    zone_map: dict[str, Zone],
    start: str,
    end: str,
    venue_id: str,
    avoid_high_density: bool,
) -> list[str] | None:
    if start not in zone_map or end not in zone_map:
        return None
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
            if avoid_high_density and neighbor_id != end:
                density, _ = store.calculate_density(venue_id, neighbor_id)
                if density >= 90:
                    continue
            visited.add(neighbor_id)
            queue.append(path + [neighbor_id])
    return None


@router.get("/{venue_id}/route", response_model=RouteResponse)
def get_route(
    venue_id: str,
    from_zone_id: str = Query(..., alias="from"),
    to_zone_id: str = Query(..., alias="to"),
) -> RouteResponse:
    if not store.get_venue(venue_id):
        raise HTTPException(status_code=404, detail="Venue not found")
    zone_map = store.get_zones(venue_id)
    if from_zone_id not in zone_map:
        raise HTTPException(status_code=404, detail="Source zone not found")
    if to_zone_id not in zone_map:
        raise HTTPException(status_code=404, detail="Destination zone not found")

    if from_zone_id == to_zone_id:
        zone = zone_map[from_zone_id]
        return RouteResponse(
            from_zone_id=from_zone_id,
            to_zone_id=to_zone_id,
            path=[from_zone_id],
            path_names=[zone.name],
            estimated_minutes=0.0,
            congestion_avoided=True,
            alternative_available=False,
            directions=[f"You are already at {zone.name}"],
        )

    if app_state.routing_service is not None:
        result = app_state.routing_service.find_route(venue_id, from_zone_id, to_zone_id)
        if result is not None:
            return RouteResponse(
                from_zone_id=from_zone_id,
                to_zone_id=to_zone_id,
                path=result["path"],
                path_names=result["zone_names"],
                estimated_minutes=result["estimated_minutes"],
                congestion_avoided=result["congestion_avoided"],
                alternative_available=result["alternative_path"] is not None,
                alternative_path=result["alternative_path"],
                directions=result["directions"],
            )

    avoidance_path = _bfs(zone_map, from_zone_id, to_zone_id, venue_id, avoid_high_density=True)
    direct_path = _bfs(zone_map, from_zone_id, to_zone_id, venue_id, avoid_high_density=False)

    if not direct_path:
        raise HTTPException(status_code=404, detail="No route found between these zones")

    if avoidance_path:
        chosen = avoidance_path
        congestion_avoided = True
        alternative_available = avoidance_path != direct_path
    else:
        chosen = direct_path
        congestion_avoided = False
        alternative_available = False

    estimated_minutes = sum(
        store.calculate_density(venue_id, zid)[0] * store.get_wait_factor(zone_map[zid].type)
        for zid in chosen
    ) + max(0, len(chosen) - 1) * 0.5

    return RouteResponse(
        from_zone_id=from_zone_id,
        to_zone_id=to_zone_id,
        path=chosen,
        path_names=[zone_map[zid].name for zid in chosen],
        estimated_minutes=round(estimated_minutes, 1),
        congestion_avoided=congestion_avoided,
        alternative_available=alternative_available,
    )


@router.get("/{venue_id}/nearest/{zone_type}")
def get_nearest(
    venue_id: str,
    zone_type: str,
    from_zone: str = Query(..., description="Starting zone ID"),
) -> dict:
    if not store.get_venue(venue_id):
        raise HTTPException(status_code=404, detail="Venue not found")
    if app_state.routing_service is None:
        raise HTTPException(status_code=503, detail="Routing service not initialized")
    result = app_state.routing_service.find_nearest(venue_id, from_zone, zone_type)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No reachable zone of type '{zone_type}' found",
        )
    return result


@router.get("/{venue_id}/exit-route")
def get_exit_route(
    venue_id: str,
    from_zone: str = Query(..., description="Starting zone ID"),
) -> dict:
    if not store.get_venue(venue_id):
        raise HTTPException(status_code=404, detail="Venue not found")
    if app_state.routing_service is None:
        raise HTTPException(status_code=503, detail="Routing service not initialized")
    result = app_state.routing_service.find_nearest(venue_id, from_zone, "exit")
    if result is None:
        raise HTTPException(status_code=404, detail="No reachable exit found")
    return result
