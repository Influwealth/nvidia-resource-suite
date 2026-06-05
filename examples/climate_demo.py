#!/usr/bin/env python3
"""Climate Demo: Earth-2 FourCastNet + CorrDiff analysis of Greenville, NC site."""
import os
from loguru import logger

from earth2.climate_sim import FourCastNetClient, CorrDiffClient, SSP_SCENARIOS
from earth2.weather import Earth2WeatherEngine

API_KEY = os.environ.get("NVIDIA_API_KEY", "")
EARTH2_API_URL = os.environ.get("EARTH2_API_URL", "")
EARTH2_API_KEY = os.environ.get("EARTH2_API_KEY", "")

# Greenville, NC coordinates (53-acre sovereign site)
GREENVILLE_LAT = 35.613
GREENVILLE_LON = -77.375

# East Flatbush, Brooklyn
EAST_FLATBUSH_LAT = 40.641
EAST_FLATBUSH_LON = -73.940


def demo_fourcastnet_forecast():
    print("\n=== FourCastNet 72-hour Forecast: Greenville, NC ===")
    fcn = FourCastNetClient(api_url=EARTH2_API_URL, api_key=EARTH2_API_KEY)
    forecast = fcn.forecast(
        lat=GREENVILLE_LAT,
        lon=GREENVILLE_LON,
        lead_hours=72,
        variables=["t2m", "u10", "v10", "tp"],
    )
    print(f"Model: {forecast['model']} | Resolution: {forecast.get('resolution_km', 25)}km")
    print(f"Synthetic mode: {forecast.get('synthetic', True)}")
    print(f"First 3 timesteps (6-hourly):")
    for step in forecast["forecast"][:3]:
        print(f"  +{step['lead_hour']}h: T={step.get('t2m', 'N/A')}°C, "
              f"Wind={step.get('u10', 0):.1f}/{step.get('v10', 0):.1f} m/s, "
              f"Precip={step.get('tp', 0)*1000:.1f}mm")


def demo_weather_engine():
    print("\n=== Earth-2 Weather Engine ===")
    fcn = FourCastNetClient(api_url=EARTH2_API_URL, api_key=EARTH2_API_KEY)
    engine = Earth2WeatherEngine(fcn)

    for loc, lat, lon in [
        ("Greenville, NC", GREENVILLE_LAT, GREENVILLE_LON),
        ("East Flatbush, Brooklyn", EAST_FLATBUSH_LAT, EAST_FLATBUSH_LON),
    ]:
        print(f"\nLocation: {loc}")
        fc = engine.forecast(location=loc, lat=lat, lon=lon, lead_hours=48)
        print(f"  Daily summaries:")
        for day in fc.daily_summary[:2]:
            print(f"    Day {day['day']}: High {day.get('t_max', 'N/A')}°C / "
                  f"Low {day.get('t_min', 'N/A')}°C, "
                  f"Precip {day.get('total_precip_mm', 0):.1f}mm")
        if fc.alerts:
            print(f"  Alerts: {fc.alerts[0]}")


def demo_climate_scenarios():
    print("\n=== IPCC Climate Scenarios: Greenville, NC ===")
    engine = Earth2WeatherEngine()
    for scenario in ["SSP1-2.6", "SSP2-4.5", "SSP5-8.5"]:
        impact = engine.greenville_climate_impact(scenario)
        sc = SSP_SCENARIOS.get(scenario)
        print(f"\n{scenario}: {sc.description if sc else ''}")
        print(f"  Temperature increase: +{impact['projected_temp_increase_c']}°C")
        print(f"  Precipitation change: {impact['precip_change_pct']}%")
        print(f"  Drought risk: {impact['drought_risk']}")
        if scenario == "SSP2-4.5":
            print(f"  Resilience actions:")
            for action in impact["resilience_actions"][:2]:
                print(f"    • {action}")


def demo_corrdiff_downscaling():
    print("\n=== CorrDiff 3km Downscaling ===")
    cdf = CorrDiffClient(api_url=EARTH2_API_URL, api_key=EARTH2_API_KEY)
    region = {
        "lat_min": 35.4, "lat_max": 35.8,
        "lon_min": -77.6, "lon_max": -77.1,
    }
    result = cdf.downscale(
        coarse_field={},  # In production: ERA5 or GCM output
        target_region=region,
        variable="precipitation",
        num_samples=2,
    )
    print(f"Model: {result['model']} | Resolution: {result['resolution_km']}km")
    print(f"Samples generated: {result['num_samples']}")
    print(f"Grid points per sample: {len(result['samples'][0])} rows x {len(result['samples'][0][0])} cols")


def demo_storm_track():
    print("\n=== Atlantic Hurricane Track Simulation ===")
    engine = Earth2WeatherEngine()
    storm = engine.track_storm(
        initial_lat=20.0,
        initial_lon=-65.0,
        storm_type="hurricane",
        lead_hours=120,
    )
    print(f"Storm type: {storm.event_type}")
    print(f"Severity: {storm.severity}")
    print(f"Max intensity: {storm.max_intensity:.0f} km/h")
    print(f"Track ({len(storm.track)} points):")
    for point in storm.track[::4]:  # Every 4th point (24h intervals)
        print(f"  +{point['t_h']}h: {point['lat']}°N {abs(point['lon'])°:.1f}°W | {point['intensity_kmh']:.0f} km/h")


if __name__ == "__main__":
    demo_fourcastnet_forecast()
    demo_weather_engine()
    demo_climate_scenarios()
    demo_corrdiff_downscaling()
    demo_storm_track()
    print("\n=== Climate demo complete! ===")
