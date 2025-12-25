#!/usr/bin/env python3
"""
MEGA EARTHQUAKE + GEOPHYSICAL DATA FETCHER
Target: 3 Million+ datapoints for heatmap/EBM/CNN training
Combines: USGS Earthquakes, NASA Solar, OpenSky Flights, Weather, Crypto (as global economic signal)
"""

import requests
import csv
import time
import os
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = "earthquake_dataset"
EARTHQUAKES_CSV = os.path.join(OUTPUT_DIR, "earthquakes.csv")
SOLAR_CSV = os.path.join(OUTPUT_DIR, "solar_events.csv")
FLIGHTS_CSV = os.path.join(OUTPUT_DIR, "flight_snapshots.csv")
WEATHER_CSV = os.path.join(OUTPUT_DIR, "weather_grid.csv")
COMBINED_CSV = os.path.join(OUTPUT_DIR, "combined_geophysical.csv")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress.json")

# Target: 3 million datapoints
TARGET_DATAPOINTS = 3_000_000

# USGS allows querying by date ranges - we'll go back 30 years
USGS_BASE = "https://earthquake.usgs.gov/fdsnws/event/1/query"

# Grid resolution for heatmaps
LAT_BINS = 180  # 1 degree resolution
LON_BINS = 360

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_dir():
    """Create output directory if needed"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_progress():
    """Load progress from previous runs"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {
        "earthquakes_fetched": 0,
        "last_earthquake_date": "1990-01-01",
        "solar_events_fetched": 0,
        "flight_snapshots": 0,
        "weather_points": 0,
        "total_datapoints": 0
    }

def save_progress(progress):
    """Save progress for resume capability"""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

def log(msg):
    """Timestamped logging"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# ============================================================================
# EARTHQUAKE DATA FETCHER (PRIMARY SOURCE - MILLIONS OF RECORDS)
# ============================================================================

EARTHQUAKE_HEADERS = [
    "id", "time", "latitude", "longitude", "depth", "magnitude", "mag_type",
    "place", "type", "status", "tsunami", "sig", "net", "nst", "dmin",
    "rms", "gap", "horizontal_error", "depth_error", "mag_error",
    "mag_nst", "year", "month", "day", "hour", "minute", "second",
    "lat_bin", "lon_bin", "energy_joules", "log_energy"
]

def magnitude_to_energy(mag):
    """Convert magnitude to energy in joules (Gutenberg-Richter)"""
    if mag is None:
        return None, None
    try:
        # log10(E) = 1.5*M + 4.8 (energy in joules)
        log_energy = 1.5 * float(mag) + 4.8
        energy = 10 ** log_energy
        return energy, log_energy
    except:
        return None, None

def parse_earthquake(feature):
    """Parse a single earthquake feature into a row"""
    props = feature.get("properties", {})
    geom = feature.get("geometry", {})
    coords = geom.get("coordinates", [None, None, None])
    
    # Extract timestamp components
    time_ms = props.get("time")
    if time_ms:
        dt = datetime.utcfromtimestamp(time_ms / 1000)
        year, month, day = dt.year, dt.month, dt.day
        hour, minute, second = dt.hour, dt.minute, dt.second
        time_str = dt.isoformat() + "Z"
    else:
        year = month = day = hour = minute = second = None
        time_str = None
    
    lat = coords[1]
    lon = coords[0]
    depth = coords[2]
    mag = props.get("mag")
    
    # Calculate grid bins for heatmap
    lat_bin = int((lat + 90) / 180 * LAT_BINS) if lat is not None else None
    lon_bin = int((lon + 180) / 360 * LON_BINS) if lon is not None else None
    
    # Clamp bins
    if lat_bin is not None:
        lat_bin = max(0, min(LAT_BINS - 1, lat_bin))
    if lon_bin is not None:
        lon_bin = max(0, min(LON_BINS - 1, lon_bin))
    
    # Calculate energy
    energy, log_energy = magnitude_to_energy(mag)
    
    return [
        feature.get("id"),
        time_str,
        lat,
        lon,
        depth,
        mag,
        props.get("magType"),
        props.get("place"),
        props.get("type"),
        props.get("status"),
        props.get("tsunami"),
        props.get("sig"),
        props.get("net"),
        props.get("nst"),
        props.get("dmin"),
        props.get("rms"),
        props.get("gap"),
        props.get("horizontalError"),
        props.get("depthError"),
        props.get("magError"),
        props.get("magNst"),
        year, month, day, hour, minute, second,
        lat_bin, lon_bin,
        energy, log_energy
    ]

def fetch_earthquakes_batch(start_date, end_date, min_magnitude=None):
    """Fetch earthquakes for a date range"""
    params = {
        "format": "geojson",
        "starttime": start_date,
        "endtime": end_date,
        "orderby": "time-asc",
        "limit": 20000  # Max allowed per request
    }
    if min_magnitude:
        params["minmagnitude"] = min_magnitude
    
    try:
        response = requests.get(USGS_BASE, params=params, timeout=60)
        if response.status_code == 200:
            data = response.json()
            features = data.get("features", [])
            return features
        else:
            log(f"  USGS returned {response.status_code} for {start_date} to {end_date}")
            return []
    except Exception as e:
        log(f"  Error fetching {start_date} to {end_date}: {e}")
        return []

def fetch_all_earthquakes(progress):
    """Fetch historical earthquakes going back decades"""
    log("=" * 70)
    log("FETCHING EARTHQUAKE DATA (1990-Present)")
    log("=" * 70)
    
    # Initialize or append to CSV
    file_exists = os.path.exists(EARTHQUAKES_CSV)
    mode = 'a' if file_exists else 'w'
    
    with open(EARTHQUAKES_CSV, mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(EARTHQUAKE_HEADERS)
        
        # Start from last progress or 1990
        start_date = datetime.strptime(progress["last_earthquake_date"], "%Y-%m-%d")
        end_date = datetime.now()
        
        # Fetch in monthly chunks (more data per request)
        current = start_date
        total_fetched = progress["earthquakes_fetched"]
        
        while current < end_date:
            chunk_end = min(current + timedelta(days=30), end_date)
            
            log(f"Fetching: {current.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")
            
            features = fetch_earthquakes_batch(
                current.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d")
            )
            
            if features:
                rows = [parse_earthquake(f) for f in features]
                writer.writerows(rows)
                f.flush()
                
                total_fetched += len(features)
                log(f"  ✅ Got {len(features)} earthquakes (Total: {total_fetched:,})")
            
            # Update progress
            progress["earthquakes_fetched"] = total_fetched
            progress["last_earthquake_date"] = chunk_end.strftime("%Y-%m-%d")
            progress["total_datapoints"] = total_fetched + progress.get("solar_events_fetched", 0) + \
                                           progress.get("flight_snapshots", 0) + progress.get("weather_points", 0)
            save_progress(progress)
            
            # Check if we've hit target
            if progress["total_datapoints"] >= TARGET_DATAPOINTS:
                log(f"🎯 Reached target of {TARGET_DATAPOINTS:,} datapoints!")
                return progress
            
            current = chunk_end
            time.sleep(0.5)  # Be nice to USGS
    
    log(f"Earthquake fetch complete: {total_fetched:,} total records")
    return progress

# ============================================================================
# SOLAR FLARE DATA (NASA DONKI)
# ============================================================================

SOLAR_HEADERS = [
    "flr_id", "begin_time", "peak_time", "end_time", "class_type",
    "source_location", "active_region", "instruments", "linked_events",
    "year", "month", "day", "hour", "class_letter", "class_number",
    "energy_estimate"
]

def parse_solar_class(class_type):
    """Parse solar flare class into letter and number"""
    if not class_type:
        return None, None, None
    try:
        letter = class_type[0]  # A, B, C, M, X
        number = float(class_type[1:])
        
        # Estimate energy based on class
        # X-class = 10^-4 W/m², M = 10^-5, C = 10^-6, B = 10^-7, A = 10^-8
        class_map = {'A': -8, 'B': -7, 'C': -6, 'M': -5, 'X': -4}
        base_power = class_map.get(letter.upper(), -6)
        energy_estimate = 10 ** base_power * number
        
        return letter, number, energy_estimate
    except:
        return None, None, None

def fetch_solar_flares(progress):
    """Fetch solar flare data from NASA DONKI"""
    log("=" * 70)
    log("FETCHING SOLAR FLARE DATA")
    log("=" * 70)
    
    file_exists = os.path.exists(SOLAR_CSV)
    mode = 'a' if file_exists else 'w'
    
    with open(SOLAR_CSV, mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(SOLAR_HEADERS)
        
        # DONKI provides ~10 years of data
        # Fetch year by year
        total_fetched = progress.get("solar_events_fetched", 0)
        
        for year in range(2010, 2026):
            start = f"{year}-01-01"
            end = f"{year}-12-31"
            
            url = f"https://api.nasa.gov/DONKI/FLR?startDate={start}&endDate={end}&api_key=DEMO_KEY"
            
            try:
                log(f"Fetching solar flares for {year}...")
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    flares = response.json()
                    
                    for flare in flares:
                        begin = flare.get("beginTime", "")
                        peak = flare.get("peakTime", "")
                        
                        # Parse time
                        try:
                            dt = datetime.fromisoformat(begin.replace("Z", ""))
                            year_val, month, day, hour = dt.year, dt.month, dt.day, dt.hour
                        except:
                            year_val = month = day = hour = None
                        
                        class_type = flare.get("classType")
                        class_letter, class_number, energy_estimate = parse_solar_class(class_type)
                        
                        instruments = ",".join([i.get("displayName", "") for i in flare.get("instruments", [])])
                        linked = ",".join([e.get("activityID", "") for e in (flare.get("linkedEvents") or [])])
                        
                        row = [
                            flare.get("flrID"),
                            begin,
                            peak,
                            flare.get("endTime"),
                            class_type,
                            flare.get("sourceLocation"),
                            flare.get("activeRegionNum"),
                            instruments,
                            linked,
                            year_val, month, day, hour,
                            class_letter, class_number,
                            energy_estimate
                        ]
                        writer.writerow(row)
                        total_fetched += 1
                    
                    f.flush()
                    log(f"  ✅ Got {len(flares)} flares for {year} (Total: {total_fetched:,})")
                    
            except Exception as e:
                log(f"  Error fetching solar data for {year}: {e}")
            
            time.sleep(1)
        
        progress["solar_events_fetched"] = total_fetched
        progress["total_datapoints"] = progress["earthquakes_fetched"] + total_fetched + \
                                       progress.get("flight_snapshots", 0) + progress.get("weather_points", 0)
        save_progress(progress)
    
    log(f"Solar flare fetch complete: {total_fetched:,} total records")
    return progress

# ============================================================================
# FLIGHT SNAPSHOTS (OpenSky - Real-time Global Coverage)
# ============================================================================

FLIGHT_HEADERS = [
    "snapshot_time", "icao24", "callsign", "origin_country",
    "longitude", "latitude", "altitude_m", "on_ground", "velocity_ms",
    "heading", "vertical_rate", "sensors", "baro_altitude",
    "squawk", "spi", "position_source",
    "lat_bin", "lon_bin", "altitude_category"
]

def categorize_altitude(alt):
    """Categorize altitude for heatmap layers"""
    if alt is None:
        return "unknown"
    elif alt < 1000:
        return "ground"
    elif alt < 5000:
        return "low"
    elif alt < 10000:
        return "medium"
    else:
        return "high"

def fetch_flight_snapshot(progress, snapshot_num):
    """Fetch a single flight snapshot"""
    try:
        response = requests.get("https://opensky-network.org/api/states/all", timeout=30)
        if response.status_code == 200:
            data = response.json()
            states = data.get("states", [])
            snapshot_time = datetime.utcfromtimestamp(data.get("time", time.time())).isoformat() + "Z"
            
            rows = []
            for state in states:
                if len(state) >= 17:
                    lat = state[6]
                    lon = state[5]
                    alt = state[7]
                    
                    # Calculate grid bins
                    lat_bin = int((lat + 90) / 180 * LAT_BINS) if lat else None
                    lon_bin = int((lon + 180) / 360 * LON_BINS) if lon else None
                    
                    if lat_bin is not None:
                        lat_bin = max(0, min(LAT_BINS - 1, lat_bin))
                    if lon_bin is not None:
                        lon_bin = max(0, min(LON_BINS - 1, lon_bin))
                    
                    rows.append([
                        snapshot_time,
                        state[0],   # icao24
                        state[1],   # callsign
                        state[2],   # origin_country
                        lon,
                        lat,
                        alt,
                        state[8],   # on_ground
                        state[9],   # velocity
                        state[10],  # heading
                        state[11],  # vertical_rate
                        state[12],  # sensors
                        state[13],  # baro_altitude
                        state[14],  # squawk
                        state[15],  # spi
                        state[16],  # position_source
                        lat_bin,
                        lon_bin,
                        categorize_altitude(alt)
                    ])
            
            return rows
        else:
            log(f"  OpenSky returned {response.status_code}")
            return []
    except Exception as e:
        log(f"  Error fetching flights: {e}")
        return []

def fetch_flight_snapshots(progress, num_snapshots=100):
    """Fetch multiple flight snapshots over time"""
    log("=" * 70)
    log(f"FETCHING {num_snapshots} FLIGHT SNAPSHOTS")
    log("=" * 70)
    
    file_exists = os.path.exists(FLIGHTS_CSV)
    mode = 'a' if file_exists else 'w'
    
    with open(FLIGHTS_CSV, mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(FLIGHT_HEADERS)
        
        total_fetched = progress.get("flight_snapshots", 0)
        
        for i in range(num_snapshots):
            # Check if we've hit target
            if progress["total_datapoints"] >= TARGET_DATAPOINTS:
                log(f"🎯 Reached target!")
                break
            
            log(f"Fetching flight snapshot {i+1}/{num_snapshots}...")
            rows = fetch_flight_snapshot(progress, i)
            
            if rows:
                writer.writerows(rows)
                f.flush()
                total_fetched += len(rows)
                log(f"  ✅ Got {len(rows)} aircraft (Total: {total_fetched:,})")
            
            progress["flight_snapshots"] = total_fetched
            progress["total_datapoints"] = progress["earthquakes_fetched"] + \
                                           progress["solar_events_fetched"] + \
                                           total_fetched + \
                                           progress.get("weather_points", 0)
            save_progress(progress)
            
            # OpenSky rate limit: 10 seconds between requests for anonymous
            time.sleep(12)
    
    log(f"Flight snapshots complete: {total_fetched:,} total records")
    return progress

# ============================================================================
# WEATHER GRID DATA (Open-Meteo - Global Grid)
# ============================================================================

WEATHER_HEADERS = [
    "latitude", "longitude", "time", "temperature_2m", "relative_humidity_2m",
    "precipitation", "surface_pressure", "wind_speed_10m", "wind_direction_10m",
    "lat_bin", "lon_bin", "fetch_time"
]

def fetch_weather_grid(progress):
    """Fetch weather data for a global grid"""
    log("=" * 70)
    log("FETCHING WEATHER GRID DATA")
    log("=" * 70)
    
    file_exists = os.path.exists(WEATHER_CSV)
    mode = 'a' if file_exists else 'w'
    
    with open(WEATHER_CSV, mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(WEATHER_HEADERS)
        
        total_fetched = progress.get("weather_points", 0)
        fetch_time = datetime.utcnow().isoformat() + "Z"
        
        # Create a grid of points (every 10 degrees for practical purposes)
        lats = range(-90, 91, 10)
        lons = range(-180, 180, 10)
        
        for lat in lats:
            for lon in lons:
                # Check target
                if progress["total_datapoints"] >= TARGET_DATAPOINTS:
                    log(f"🎯 Reached target!")
                    return progress
                
                try:
                    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,surface_pressure,wind_speed_10m,wind_direction_10m"
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        current = data.get("current", {})
                        
                        lat_bin = int((lat + 90) / 180 * LAT_BINS)
                        lon_bin = int((lon + 180) / 360 * LON_BINS)
                        
                        row = [
                            lat, lon,
                            current.get("time"),
                            current.get("temperature_2m"),
                            current.get("relative_humidity_2m"),
                            current.get("precipitation"),
                            current.get("surface_pressure"),
                            current.get("wind_speed_10m"),
                            current.get("wind_direction_10m"),
                            lat_bin, lon_bin,
                            fetch_time
                        ]
                        writer.writerow(row)
                        total_fetched += 1
                        
                        if total_fetched % 50 == 0:
                            log(f"  Weather points: {total_fetched:,}")
                            f.flush()
                    
                except Exception as e:
                    pass  # Skip failed points silently
                
                time.sleep(0.1)  # Rate limiting
        
        progress["weather_points"] = total_fetched
        progress["total_datapoints"] = progress["earthquakes_fetched"] + \
                                       progress["solar_events_fetched"] + \
                                       progress["flight_snapshots"] + \
                                       total_fetched
        save_progress(progress)
    
    log(f"Weather grid complete: {total_fetched:,} total records")
    return progress

# ============================================================================
# COMBINED DATASET BUILDER
# ============================================================================

def create_combined_dataset(progress):
    """Create a combined dataset with all sources aligned by lat/lon bins"""
    log("=" * 70)
    log("CREATING COMBINED HEATMAP-READY DATASET")
    log("=" * 70)
    
    # This creates a summary by grid cell - perfect for heatmap CNNs
    
    # First, aggregate earthquakes by grid cell
    earthquake_grid = {}
    if os.path.exists(EARTHQUAKES_CSV):
        with open(EARTHQUAKES_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                lat_bin = row.get("lat_bin")
                lon_bin = row.get("lon_bin")
                if lat_bin and lon_bin:
                    key = (int(lat_bin), int(lon_bin))
                    if key not in earthquake_grid:
                        earthquake_grid[key] = {
                            "count": 0,
                            "total_magnitude": 0,
                            "max_magnitude": 0,
                            "total_energy": 0
                        }
                    earthquake_grid[key]["count"] += 1
                    mag = float(row.get("magnitude") or 0)
                    earthquake_grid[key]["total_magnitude"] += mag
                    earthquake_grid[key]["max_magnitude"] = max(earthquake_grid[key]["max_magnitude"], mag)
                    energy = float(row.get("energy_joules") or 0)
                    earthquake_grid[key]["total_energy"] += energy
    
    log(f"  Aggregated {len(earthquake_grid)} earthquake grid cells")
    
    # Aggregate flights by grid cell
    flight_grid = {}
    if os.path.exists(FLIGHTS_CSV):
        with open(FLIGHTS_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                lat_bin = row.get("lat_bin")
                lon_bin = row.get("lon_bin")
                if lat_bin and lon_bin:
                    key = (int(lat_bin), int(lon_bin))
                    if key not in flight_grid:
                        flight_grid[key] = {"count": 0, "total_altitude": 0}
                    flight_grid[key]["count"] += 1
                    alt = float(row.get("altitude_m") or 0)
                    flight_grid[key]["total_altitude"] += alt
    
    log(f"  Aggregated {len(flight_grid)} flight grid cells")
    
    # Create combined output
    combined_headers = [
        "lat_bin", "lon_bin", "center_lat", "center_lon",
        "earthquake_count", "total_magnitude", "avg_magnitude", "max_magnitude",
        "total_energy_joules", "log_total_energy",
        "flight_count", "avg_altitude",
        "seismic_risk_score", "activity_score"
    ]
    
    with open(COMBINED_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(combined_headers)
        
        # Create all grid cells
        all_keys = set(earthquake_grid.keys()) | set(flight_grid.keys())
        
        for lat_bin, lon_bin in all_keys:
            # Calculate center coordinates
            center_lat = (lat_bin / LAT_BINS * 180) - 90 + (180 / LAT_BINS / 2)
            center_lon = (lon_bin / LON_BINS * 360) - 180 + (360 / LON_BINS / 2)
            
            # Earthquake stats
            eq = earthquake_grid.get((lat_bin, lon_bin), {})
            eq_count = eq.get("count", 0)
            total_mag = eq.get("total_magnitude", 0)
            avg_mag = total_mag / eq_count if eq_count > 0 else 0
            max_mag = eq.get("max_magnitude", 0)
            total_energy = eq.get("total_energy", 0)
            log_energy = np.log10(total_energy) if total_energy > 0 else 0
            
            # Flight stats
            fl = flight_grid.get((lat_bin, lon_bin), {})
            flight_count = fl.get("count", 0)
            avg_alt = fl.get("total_altitude", 0) / flight_count if flight_count > 0 else 0
            
            # Derived scores for ML
            seismic_risk = min(100, eq_count * avg_mag) if eq_count > 0 else 0
            activity_score = eq_count + flight_count
            
            writer.writerow([
                lat_bin, lon_bin, round(center_lat, 2), round(center_lon, 2),
                eq_count, round(total_mag, 2), round(avg_mag, 2), round(max_mag, 2),
                total_energy, round(log_energy, 2),
                flight_count, round(avg_alt, 2),
                round(seismic_risk, 2), activity_score
            ])
        
        log(f"  Combined dataset: {len(all_keys)} grid cells")
    
    return progress

# ============================================================================
# EXTENDED EARTHQUAKE FETCH (SMALLER MAGNITUDES FOR MORE DATA)
# ============================================================================

def fetch_micro_earthquakes(progress):
    """Fetch micro-earthquakes (mag < 2) for more data density"""
    log("=" * 70)
    log("FETCHING MICRO-EARTHQUAKES (All Magnitudes)")
    log("=" * 70)
    
    # USGS has TONS of small earthquakes we can grab
    # Focus on recent years with better detection
    
    with open(EARTHQUAKES_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        total_fetched = progress["earthquakes_fetched"]
        
        # Fetch by week for recent years (more granular = more data)
        start = datetime(2020, 1, 1)
        end = datetime.now()
        current = start
        
        while current < end and progress["total_datapoints"] < TARGET_DATAPOINTS:
            chunk_end = min(current + timedelta(days=7), end)
            
            log(f"Fetching micro-quakes: {current.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")
            
            # No magnitude filter = get EVERYTHING
            features = fetch_earthquakes_batch(
                current.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
                min_magnitude=None
            )
            
            if features:
                rows = [parse_earthquake(f) for f in features]
                writer.writerows(rows)
                f.flush()
                
                total_fetched += len(features)
                log(f"  ✅ Got {len(features)} quakes (Total: {total_fetched:,})")
            
            progress["earthquakes_fetched"] = total_fetched
            progress["total_datapoints"] = total_fetched + progress["solar_events_fetched"] + \
                                           progress["flight_snapshots"] + progress["weather_points"]
            save_progress(progress)
            
            current = chunk_end
            time.sleep(0.3)
    
    return progress

# ============================================================================
# STATS AND SUMMARY
# ============================================================================

def print_final_stats(progress):
    """Print final statistics"""
    log("=" * 70)
    log("📊 FINAL DATASET STATISTICS")
    log("=" * 70)
    
    total = progress["total_datapoints"]
    target_pct = (total / TARGET_DATAPOINTS) * 100
    
    print(f"""
    ┌─────────────────────────────────────────────────────────────┐
    │  EARTHQUAKE + GEOPHYSICAL MEGA DATASET                     │
    ├─────────────────────────────────────────────────────────────┤
    │  🌍 Earthquakes:      {progress['earthquakes_fetched']:>12,} records            │
    │  ☀️  Solar Flares:     {progress['solar_events_fetched']:>12,} records            │
    │  ✈️  Flight Points:    {progress['flight_snapshots']:>12,} records            │
    │  🌡️  Weather Points:   {progress['weather_points']:>12,} records            │
    ├─────────────────────────────────────────────────────────────┤
    │  📈 TOTAL DATAPOINTS: {total:>12,}                     │
    │  🎯 Target Progress:  {target_pct:>11.1f}%                      │
    └─────────────────────────────────────────────────────────────┘
    
    📁 Output Files:
       • {EARTHQUAKES_CSV}
       • {SOLAR_CSV}
       • {FLIGHTS_CSV}
       • {WEATHER_CSV}
       • {COMBINED_CSV}
    """)
    
    # File sizes
    for filepath in [EARTHQUAKES_CSV, SOLAR_CSV, FLIGHTS_CSV, WEATHER_CSV, COMBINED_CSV]:
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"       {os.path.basename(filepath)}: {size_mb:.1f} MB")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline"""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  🌍 MEGA EARTHQUAKE + GEOPHYSICAL DATA FETCHER                   ║
    ║  Target: 3,000,000 datapoints for Heatmap/EBM/CNN Training       ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    ensure_dir()
    progress = load_progress()
    
    log(f"Starting from: {progress['total_datapoints']:,} existing datapoints")
    
    try:
        # Need numpy for log calculations
        global np
        import numpy as np
    except ImportError:
        log("Installing numpy...")
        os.system("pip install numpy")
        import numpy as np
    
    # Phase 1: Historical Earthquakes (30+ years = ~2M records)
    if progress["total_datapoints"] < TARGET_DATAPOINTS:
        progress = fetch_all_earthquakes(progress)
    
    # Phase 2: Micro-earthquakes for density
    if progress["total_datapoints"] < TARGET_DATAPOINTS:
        progress = fetch_micro_earthquakes(progress)
    
    # Phase 3: Solar Flares (~10k records)
    if progress["total_datapoints"] < TARGET_DATAPOINTS:
        progress = fetch_solar_flares(progress)
    
    # Phase 4: Flight Snapshots (each = ~10k aircraft)
    if progress["total_datapoints"] < TARGET_DATAPOINTS:
        # Calculate how many snapshots we need
        remaining = TARGET_DATAPOINTS - progress["total_datapoints"]
        snapshots_needed = (remaining // 8000) + 1  # ~8k aircraft per snapshot
        progress = fetch_flight_snapshots(progress, min(snapshots_needed, 500))
    
    # Phase 5: Weather Grid
    if progress["total_datapoints"] < TARGET_DATAPOINTS:
        progress = fetch_weather_grid(progress)
    
    # Phase 6: Create combined heatmap-ready dataset
    progress = create_combined_dataset(progress)
    
    # Final stats
    print_final_stats(progress)
    
    log("✅ Data collection complete!")
    log("Next: Use earthquakes.csv directly or combined_geophysical.csv for heatmap generation")

if __name__ == "__main__":
    main()