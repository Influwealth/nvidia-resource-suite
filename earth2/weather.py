"""Earth-2 weather engine: short-range prediction, storm tracking, extreme events."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class WeatherForecast:
    location: str
    lat: float
    lon: float
    model: str
    generated_at: float
    hourly: list[dict] = field(default_factory=list)  # 6-hourly steps
    daily_summary: list[dict] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)
    confidence: float = 0.85


@dataclass
class ExtremeEvent:
    event_type: str   # hurricane | tornado | flood | drought | heatwave | blizzard
    location: str
    lat: float
    lon: float
    severity: str     # watch | warning | emergency
    start_time: float
    end_time: float | None
    max_intensity: float
    probability: float
    track: list[dict] = field(default_factory=list)  # [{"lat", "lon", "t", "intensity"}]


class Earth2WeatherEngine:
    """Short-range weather prediction and hazard detection engine.

    Integrates FourCastNet forecasts with post-processing for
    actionable weather intelligence.
    """

    def __init__(self, fourcastnet: Any | None = None, corrdiff: Any | None = None):
        self._fcn = fourcastnet
        self._cdf = corrdiff

    def forecast(
        self,
        location: str,
        lat: float,
        lon: float,
        lead_hours: int = 72,
    ) -> WeatherForecast:
        if self._fcn:
            raw = self._fcn.forecast(lat, lon, lead_hours=lead_hours)
        else:
            raw = self._synthetic_raw(lat, lon, lead_hours)

        hourly = raw.get("forecast", [])
        daily = self._aggregate_daily(hourly)
        alerts = self._check_alerts(hourly)

        return WeatherForecast(
            location=location, lat=lat, lon=lon,
            model=raw.get("model", "FourCastNet-v2"),
            generated_at=time.time(),
            hourly=hourly, daily_summary=daily, alerts=alerts,
        )

    def _synthetic_raw(self, lat: float, lon: float, lead_hours: int) -> dict:
        import random
        steps = lead_hours // 6
        base_temp = 15.0 + 10.0 * math.sin(math.radians(lat))
        forecast = []
        for step in range(steps):
            t = step * 6
            forecast.append({
                "lead_hour": t,
                "t2m": round(base_temp + 5.0 * math.sin(math.pi * (t % 24) / 12) + random.gauss(0, 2), 2),
                "u10": round(random.gauss(0, 8), 2),
                "v10": round(random.gauss(0, 8), 2),
                "tp": round(max(0, random.gauss(0.3, 0.8)), 3),
                "sp": round(101325 + random.gauss(0, 300), 1),
                "r2": round(min(100, max(10, 65 + random.gauss(0, 15))), 1),
            })
        return {"forecast": forecast, "model": "FourCastNet-synthetic"}

    def _aggregate_daily(self, hourly: list[dict]) -> list[dict]:
        days: dict[int, list[dict]] = {}
        for step in hourly:
            day = step["lead_hour"] // 24
            days.setdefault(day, []).append(step)
        daily = []
        for day_idx in sorted(days):
            steps = days[day_idx]
            temps = [s["t2m"] for s in steps if "t2m" in s]
            precip = sum(s.get("tp", 0) for s in steps)
            daily.append({
                "day": day_idx,
                "t_max": round(max(temps), 2) if temps else None,
                "t_min": round(min(temps), 2) if temps else None,
                "t_mean": round(sum(temps) / len(temps), 2) if temps else None,
                "total_precip_mm": round(precip * 1000, 2),
            })
        return daily

    def _check_alerts(self, hourly: list[dict]) -> list[str]:
        alerts = []
        for step in hourly:
            wind = math.sqrt(step.get("u10", 0)**2 + step.get("v10", 0)**2)
            temp = step.get("t2m", 15)
            precip = step.get("tp", 0)
            if wind > 20:
                alerts.append(f"Wind advisory: {wind:.1f} m/s at lead hour {step['lead_hour']}")
            if temp > 35:
                alerts.append(f"Heat warning: {temp}°C at lead hour {step['lead_hour']}")
            if temp < -20:
                alerts.append(f"Extreme cold: {temp}°C at lead hour {step['lead_hour']}")
            if precip > 0.05:
                alerts.append(f"Heavy precipitation: {precip*1000:.1f}mm at lead hour {step['lead_hour']}")
        return list(set(alerts))[:5]  # top 5 unique alerts

    def track_storm(
        self,
        initial_lat: float,
        initial_lon: float,
        storm_type: str = "hurricane",
        lead_hours: int = 120,
    ) -> ExtremeEvent:
        """Generate a synthetic storm track for educational simulation."""
        import random
        track = []
        lat, lon = initial_lat, initial_lon
        intensity = 120.0  # km/h initial wind speed
        for h in range(0, lead_hours, 6):
            track.append({"lat": round(lat, 3), "lon": round(lon, 3), "t_h": h, "intensity_kmh": round(intensity, 1)})
            lat += random.gauss(0.5, 0.2)   # northward drift
            lon += random.gauss(-0.8, 0.3)  # westward drift
            intensity += random.gauss(-2, 5) if h < 60 else random.gauss(-8, 3)
            intensity = max(0, intensity)

        severity = "emergency" if max(t["intensity_kmh"] for t in track) > 180 else "warning"
        return ExtremeEvent(
            event_type=storm_type,
            location=f"{initial_lat:.1f}N {abs(initial_lon):.1f}W",
            lat=initial_lat, lon=initial_lon,
            severity=severity,
            start_time=time.time(),
            end_time=None,
            max_intensity=max(t["intensity_kmh"] for t in track),
            probability=0.75,
            track=track,
        )

    def greenville_climate_impact(
        self,
        scenario: str = "SSP2-4.5",
    ) -> dict:
        """Climate impact projection for the Greenville, NC 53-acre site."""
        scenarios = {
            "SSP1-2.6": {"temp_delta": 1.2, "precip_change_pct": 5, "drought_risk": "low"},
            "SSP2-4.5": {"temp_delta": 2.1, "precip_change_pct": -8, "drought_risk": "moderate"},
            "SSP5-8.5": {"temp_delta": 4.0, "precip_change_pct": -18, "drought_risk": "high"},
        }
        impact = scenarios.get(scenario, scenarios["SSP2-4.5"])
        return {
            "site": "Greenville Sovereign Site, NC (35.61°N, 77.37°W)",
            "scenario": scenario,
            "projected_temp_increase_c": impact["temp_delta"],
            "precip_change_pct": impact["precip_change_pct"],
            "drought_risk": impact["drought_risk"],
            "agricultural_note": "Irrigation planning should account for 15-25% longer dry spells under SSP2-4.5+",
            "resilience_actions": [
                "Install rainwater harvesting cisterns",
                "Plant drought-tolerant cover crops",
                "Deploy soil moisture sensors with Warp-based irrigation control",
                "Install reflective roofing to offset urban heat island",
            ],
        }
