from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from models.venue import Alert
from state import store

router = APIRouter(prefix="/venues", tags=["alerts"])


@router.get("/{venue_id}/alerts", response_model=list[Alert])
def get_alerts(venue_id: str) -> list[Alert]:
    if not store.get_venue(venue_id):
        raise HTTPException(status_code=404, detail="Venue not found")
    store.generate_alerts(venue_id)
    venue_alerts = store.alerts.get(venue_id, {})
    return [a for a in venue_alerts.values() if not a.resolved]


@router.post("/{venue_id}/alerts/{alert_id}/resolve")
def resolve_alert(venue_id: str, alert_id: str) -> dict:
    if not store.get_venue(venue_id):
        raise HTTPException(status_code=404, detail="Venue not found")
    venue_alerts = store.alerts.get(venue_id, {})
    alert = venue_alerts.get(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    if alert.resolved:
        return {"message": "Alert already resolved", "alert_id": alert_id}
    alert.resolved = True
    alert.resolved_at = datetime.now(timezone.utc)
    return {"message": "Alert resolved", "alert_id": alert_id}
