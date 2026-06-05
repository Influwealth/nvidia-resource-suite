"""Earth-2: CorrDiff downscaling, FourCastNet global forecasting, climate scenarios."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from loguru import logger

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class ClimateScenario:
    """IPCC scenario specification."""
    name: str          # SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-8.5
    co2_ppm: float    # atmospheric CO2 concentration
    temp_delta_c: float  # global mean temperature change vs 1850-1900
    year: int
    description: str = ""


SSP_SCENARIOS = {
    "SSP1-2.6": ClimateScenario("SSP1-2.6", 443, 1.8, 2100, "Sustainable development, net-zero by 2050"),
    "SSP2-4.5": ClimateScenario("SSP2-4.5", 538, 2.7, 2100, "Intermediate emissions, current policies"),
    "SSP3-7.0": ClimateScenario("SSP3-7.0", 670, 3.6, 2100, "High emissions, regional rivalry"),
    "SSP5-8.5": ClimateScenario("SSP5-8.5", 936, 4.4, 2100, "Very high emissions, fossil-fueled growth"),
}


class FourCastNetClient:
    """NVIDIA FourCastNet global atmospheric forecast model.

    FourCastNet uses adaptive Fourier neural operators (AFNO) to produce
    global weather forecasts at 25km resolution in seconds on a single GPU.
    Connects to Earth-2 API or runs locally with pretrained weights.
    """

    def __init__(self, api_url: str | None = None, api_key: str | None = None):
        self.api_url = api_url or os.environ.get("EARTH2_API_URL", "")
        self.api_key = api_key or os.environ.get("EARTH2_API_KEY", "")
        self._http = httpx.Client(timeout=120.0)
        self._available = bool(self.api_url and self.api_key)
        if not self._available:
            logger.warning("Earth-2 API not configured — FourCastNet in synthetic mode")

    def forecast(
        self,
        lat: float,
        lon: float,
        lead_hours: int = 72,
        variables: list[str] | None = None,
    ) -> dict:
        """Generate a weather forecast for a lat/lon point.

        Args:
            lat: Latitude (-90 to 90)
            lon: Longitude (-180 to 180)
            lead_hours: Forecast horizon in hours (max 240 for 10-day)
            variables: ERA5 variable names e.g. ["u10", "v10", "t2m", "sp"]
        """
        variables = variables or ["t2m", "u10", "v10", "tp", "sp", "r2"]
        if self._available:
            return self._api_forecast(lat, lon, lead_hours, variables)
        return self._synthetic_forecast(lat, lon, lead_hours, variables)

    def _api_forecast(self, lat: float, lon: float, lead_hours: int, variables: list[str]) -> dict:
        r = self._http.post(
            f"{self.api_url}/forecast",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"lat": lat, "lon": lon, "lead_hours": lead_hours, "variables": variables},
        )
        r.raise_for_status()
        return r.json()

    def _synthetic_forecast(self, lat: float, lon: float, lead_hours: int, variables: list[str]) -> dict:
        import math, random
        steps = lead_hours // 6  # 6-hour intervals
        forecast = []
        for step in range(steps):
            t = step * 6
            entry = {"lead_hour": t, "lat": lat, "lon": lon}
            # Synthetic seasonal/diurnal signal
            base_temp = 15.0 + 10.0 * math.sin(math.radians(lat))
            diurnal = 5.0 * math.sin(math.pi * (t % 24) / 12)
            entry["t2m"] = round(base_temp + diurnal + random.gauss(0, 1.5), 2)
            entry["u10"] = round(random.gauss(0, 5), 2)
            entry["v10"] = round(random.gauss(0, 5), 2)
            entry["tp"] = round(max(0, random.gauss(0.5, 1.0)), 3)
            entry["sp"] = round(101325 + random.gauss(0, 500), 1)
            entry["r2"] = round(min(100, max(0, 65 + random.gauss(0, 10))), 1)
            forecast.append({k: v for k, v in entry.items() if k in ["lead_hour", "lat", "lon"] + variables})
        return {
            "model": "FourCastNet-v2",
            "lat": lat, "lon": lon,
            "lead_hours": lead_hours,
            "resolution_km": 25,
            "variables": variables,
            "forecast": forecast,
            "synthetic": not self._available,
        }

    def global_ensemble(
        self,
        lat: float,
        lon: float,
        lead_hours: int = 120,
        ensemble_size: int = 10,
    ) -> dict:
        """Multi-member ensemble for probabilistic forecasting."""
        import random
        members = []
        for member_id in range(ensemble_size):
            fc = self._synthetic_forecast(lat, lon, lead_hours, ["t2m", "tp"])
            fc["member"] = member_id
            members.append(fc)
        return {
            "ensemble_size": ensemble_size,
            "lat": lat, "lon": lon,
            "members": members,
        }


class CorrDiffClient:
    """NVIDIA CorrDiff statistical downscaling model.

    CorrDiffuses low-resolution GCM output to high-resolution (3km)
    regional climate projections using diffusion models trained on ERA5.
    """

    def __init__(self, api_url: str | None = None, api_key: str | None = None):
        self.api_url = api_url or os.environ.get("EARTH2_API_URL", "")
        self.api_key = api_key or os.environ.get("EARTH2_API_KEY", "")
        self._http = httpx.Client(timeout=300.0)
        self._available = bool(self.api_url and self.api_key)

    def downscale(
        self,
        coarse_field: dict,  # low-res field data
        target_region: dict,  # {lat_min, lat_max, lon_min, lon_max}
        variable: str = "precipitation",
        num_samples: int = 4,
    ) -> dict:
        """Downscale a coarse climate field to 3km resolution.

        Returns ensemble of high-res realizations from the diffusion model.
        """
        if self._available:
            r = self._http.post(
                f"{self.api_url}/downscale",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"coarse_field": coarse_field, "target_region": target_region,
                      "variable": variable, "num_samples": num_samples},
            )
            r.raise_for_status()
            return r.json()

        return self._synthetic_downscale(target_region, variable, num_samples)

    def _synthetic_downscale(self, region: dict, variable: str, num_samples: int) -> dict:
        import random
        lat_pts = 20
        lon_pts = 20
        samples = []
        for s in range(num_samples):
            grid = []
            for i in range(lat_pts):
                row = []
                lat = region.get("lat_min", 0) + i * (region.get("lat_max", 1) - region.get("lat_min", 0)) / lat_pts
                for j in range(lon_pts):
                    lon = region.get("lon_min", 0) + j * (region.get("lon_max", 1) - region.get("lon_min", 0)) / lon_pts
                    row.append({"lat": round(lat, 4), "lon": round(lon, 4), variable: round(max(0, random.gauss(3.0, 2.0)), 3)})
                grid.append(row)
            samples.append(grid)
        return {
            "model": "CorrDiff",
            "variable": variable,
            "resolution_km": 3,
            "num_samples": num_samples,
            "region": region,
            "samples": samples,
            "synthetic": not self._available,
        }

    def compare_scenarios(
        self,
        region: dict,
        variable: str = "temperature",
        scenarios: list[str] | None = None,
    ) -> dict:
        """Compare climate change scenarios for a region."""
        scenarios = scenarios or ["SSP1-2.6", "SSP2-4.5", "SSP5-8.5"]
        results = {}
        for ssp in scenarios:
            scenario = SSP_SCENARIOS.get(ssp)
            if scenario:
                baseline = self.downscale({}, region, variable, num_samples=1)
                baseline["scenario"] = ssp
                baseline["temp_delta_c"] = scenario.temp_delta_c
                baseline["description"] = scenario.description
                results[ssp] = baseline
        return {
            "region": region,
            "variable": variable,
            "scenario_comparison": results,
            "reference": "IPCC AR6",
        }
