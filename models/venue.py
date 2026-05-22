import uuid
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field


def _new_id() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class VenueCreate(BaseModel):
    name: str
    type: str
    address: str


class Venue(BaseModel):
    id: str = Field(default_factory=_new_id)
    name: str
    type: str
    address: str
    created_at: datetime = Field(default_factory=_now)


class ZoneCreate(BaseModel):
    name: str
    type: str
    area_m2: float
    x_pct: float
    y_pct: float
    w_pct: float
    h_pct: float
    max_capacity: int
    adjacent_zone_ids: list[str] = Field(default_factory=list)


class ZoneUpdate(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    area_m2: Optional[float] = None
    x_pct: Optional[float] = None
    y_pct: Optional[float] = None
    w_pct: Optional[float] = None
    h_pct: Optional[float] = None
    max_capacity: Optional[int] = None
    adjacent_zone_ids: Optional[list[str]] = None


class Zone(BaseModel):
    id: str = Field(default_factory=_new_id)
    venue_id: str
    name: str
    type: str
    area_m2: float
    x_pct: float
    y_pct: float
    w_pct: float
    h_pct: float
    max_capacity: int
    adjacent_zone_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)


class VenueWithZones(Venue):
    zones: list[Zone] = Field(default_factory=list)


class CameraIngest(BaseModel):
    venue_id: str
    zone_id: str
    person_count: int
    timestamp: datetime
    camera_id: str


class WifiIngest(BaseModel):
    venue_id: str
    zone_id: str
    device_count: int
    timestamp: datetime


class ManualIngest(BaseModel):
    venue_id: str
    zone_id: str
    person_count: int
    timestamp: datetime
    staff_id: Optional[str] = None
    notes: Optional[str] = None


class DensityReading(BaseModel):
    id: str = Field(default_factory=_new_id)
    venue_id: str
    zone_id: str
    person_count: int
    timestamp: datetime
    source: str
    camera_id: Optional[str] = None


class ZoneDensity(BaseModel):
    zone_id: str
    zone_name: str
    zone_type: str
    density_score: float
    risk_level: str
    person_count: int
    wait_minutes: float
    max_capacity: int
    prediction_confidence: float = 1.0


class WaitTime(BaseModel):
    zone_id: str
    zone_name: str
    zone_type: str
    density_score: float
    wait_minutes: float


class WaitTimePrediction(BaseModel):
    zone_type: str
    current_density: float
    hour: int
    predicted_wait_minutes: float
    confidence_low: float
    confidence_high: float
    model_trained: bool


class Alert(BaseModel):
    id: str = Field(default_factory=_new_id)
    venue_id: str
    zone_id: str
    zone_name: str
    level: str
    density_score: float
    message: str
    created_at: datetime = Field(default_factory=_now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class RouteResponse(BaseModel):
    from_zone_id: str
    to_zone_id: str
    path: list[str]
    path_names: list[str]
    estimated_minutes: float
    congestion_avoided: bool
    alternative_available: bool
    alternative_path: list[str] | None = None
    directions: list[str] = Field(default_factory=list)


class AnalyticsSummary(BaseModel):
    venue_id: str
    venue_name: str
    venue_type: str
    total_zones: int
    total_readings_today: int
    average_density: float
    active_alerts: int
    peak_zones: list[ZoneDensity]
