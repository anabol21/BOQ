from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


ShowFormat = Literal["2D", "3D", "IMAX", "4DX", "OTHER"]
HourBucket = Literal["morning", "day", "evening", "late"]
RuntimeBucket = Literal["short", "medium", "long"]


class ShowRecord(BaseModel):
    show_id: str = Field(..., min_length=1)
    cinema_id: str = Field(..., min_length=1)
    hall_id: str = Field(..., min_length=1)
    hall_capacity: int = Field(..., gt=0)
    movie_id: str = Field(..., min_length=1)
    genre: str = Field(..., min_length=1)
    age_rating: str = Field(..., min_length=1)
    runtime_min: int = Field(..., gt=0)
    release_date: date
    show_datetime: datetime
    format: ShowFormat
    base_price: float = Field(..., ge=0)
    sold_tickets: int = Field(..., ge=0)
    occupancy_rate: float = Field(..., ge=0.0, le=1.0)

    city: Optional[str] = None
    day_of_week: Optional[int] = Field(default=None, ge=0, le=6)
    is_weekend: Optional[bool] = None
    hour_bucket: Optional[HourBucket] = None
    month: Optional[int] = Field(default=None, ge=1, le=12)
    holiday_flag: Optional[bool] = None
    release_age_days: Optional[int] = Field(default=None, ge=0)
    lead_time_days: Optional[float] = Field(default=None, ge=0.0)
    prime_time_flag: Optional[bool] = None
    runtime_bucket: Optional[RuntimeBucket] = None
    screening_number_for_movie: Optional[int] = Field(default=None, ge=0)

    @field_validator("genre", "age_rating", "city")
    @classmethod
    def strip_strings(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        value = value.strip()
        return value or None

    @model_validator(mode="after")
    def check_consistency(self) -> "ShowRecord":
        if self.release_date > self.show_datetime.date():
            raise ValueError("release_date cannot be later than show_datetime")
        if self.sold_tickets > self.hall_capacity:
            raise ValueError("sold_tickets cannot exceed hall_capacity")

        expected_occupancy = self.sold_tickets / self.hall_capacity
        if abs(expected_occupancy - self.occupancy_rate) > 1e-3:
            raise ValueError("occupancy_rate must be consistent with sold_tickets / hall_capacity")

        return self
