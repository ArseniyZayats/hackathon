"""
parser.py — Ardupilot DataFlash (.bin) log parser
Extracts GPS and IMU messages into pandas DataFrames.

Ardupilot DataFlash format:
  - Each record starts with 0xA3 0x95 header bytes
  - Followed by message type byte, then payload
  - Format definitions are stored in FMT messages
"""

import struct
import numpy as np
import pandas as pd
from pymavlink import mavutil


def parse_bin(filepath: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse an Ardupilot .bin DataFlash log file.
    Returns (gps_df, imu_df) as normalized DataFrames.
    """
    log = mavutil.mavlink_connection(filepath, dialect="ardupilotmega")

    gps_raw = []
    imu_raw = []

    while True:
        msg = log.recv_match(type=["GPS", "GPS2", "GPSB", "IMU", "IMU2"])
        if msg is None:
            break

        mtype = msg.get_type()

        if mtype in ("GPS", "GPS2", "GPSB"):
            entry = {
                "timestamp": getattr(msg, "TimeUS", 0),
                "lat": getattr(msg, "Lat", None),
                "lon": getattr(msg, "Lng", getattr(msg, "Lon", None)),
                "alt": getattr(msg, "Alt", None),
                "speed": getattr(msg, "Spd", None),
                "num_sats": getattr(msg, "NSats", None),
                "status": getattr(msg, "Status", None),
            }
            if entry["lat"] is not None:
                gps_raw.append(entry)

        elif mtype in ("IMU", "IMU2"):
            entry = {
                "timestamp": getattr(msg, "TimeUS", 0),
                "accX": getattr(msg, "AccX", None),
                "accY": getattr(msg, "AccY", None),
                "accZ": getattr(msg, "AccZ", None),
                "gyrX": getattr(msg, "GyrX", None),
                "gyrY": getattr(msg, "GyrY", None),
                "gyrZ": getattr(msg, "GyrZ", None),
            }
            imu_raw.append(entry)

    if not gps_raw and not imu_raw:
        raise ValueError(
            "No GPS or IMU messages found. "
            "File may be corrupted or not an Ardupilot DataFlash log."
        )

    gps_df = _normalize_gps(pd.DataFrame(gps_raw)) if gps_raw else pd.DataFrame()
    imu_df = _normalize_imu(pd.DataFrame(imu_raw)) if imu_raw else pd.DataFrame()

    return gps_df, imu_df


def _normalize_gps(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw GPS fields to standard units."""
    df = df.copy()
    # TimeUS → seconds
    df["timestamp"] = df["timestamp"] / 1e6

    # Ardupilot stores lat/lon as integer degrees * 1e7
    if df["lat"].abs().max() > 1000:
        df["lat"] = df["lat"] / 1e7
        df["lon"] = df["lon"] / 1e7

    # Alt in cm → meters (some firmware versions)
    if df["alt"].abs().max() > 10_000:
        df["alt"] = df["alt"] / 100.0

    df = df.dropna(subset=["lat", "lon"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _normalize_imu(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw IMU fields to standard units."""
    df = df.copy()
    df["timestamp"] = df["timestamp"] / 1e6
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def sampling_frequencies(gps_df: pd.DataFrame, imu_df: pd.DataFrame) -> dict:
    """Calculate approximate sensor sampling frequencies (Hz)."""
    result = {}
    if not gps_df.empty:
        gps_dt = gps_df["timestamp"].diff().median()
        result["gps_hz"] = round(1.0 / gps_dt, 2) if gps_dt > 0 else 0
        result["gps_count"] = len(gps_df)
    if not imu_df.empty:
        imu_dt = imu_df["timestamp"].diff().median()
        result["imu_hz"] = round(1.0 / imu_dt, 2) if imu_dt > 0 else 0
        result["imu_count"] = len(imu_df)
    return result


# ---------------------------------------------------------------------------
# Synthetic data generator — used for demo when no real .bin file is provided
# ---------------------------------------------------------------------------

def generate_synthetic_flight(
    duration_s: float = 120.0,
    gps_hz: float = 5.0,
    imu_hz: float = 50.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a realistic synthetic UAV flight dataset for demonstration.
    Flight profile: takeoff → cruise → descent → landing.
    """
    rng = np.random.default_rng(42)

    # ----- GPS track -----
    n_gps = int(duration_s * gps_hz)
    t_gps = np.linspace(0, duration_s, n_gps)

    # Spiral trajectory over Lviv (for demo)
    lat0, lon0 = 49.8397, 24.0297
    radius_deg = 0.002  # ~200 m

    phase = t_gps / duration_s * 4 * np.pi  # 2 full circles
    lat = lat0 + radius_deg * np.sin(phase) * (t_gps / duration_s)
    lon = lon0 + radius_deg * np.cos(phase) * (t_gps / duration_s)

    # Altitude: takeoff 0→50m, cruise 50m, descent 50→0m
    alt_profile = np.concatenate([
        np.linspace(0, 50, n_gps // 4),
        np.full(n_gps // 2, 50) + rng.normal(0, 0.5, n_gps // 2),
        np.linspace(50, 0, n_gps - 3 * (n_gps // 4)),
    ])
    alt = alt_profile[:n_gps] + rng.normal(0, 0.2, n_gps)
    alt = np.clip(alt, 0, None)

    # Horizontal speed from differentiation
    dx = np.gradient(lon, t_gps) * 111_320 * np.cos(np.radians(lat0))
    dy = np.gradient(lat, t_gps) * 110_540
    speed = np.sqrt(dx**2 + dy**2)

    gps_df = pd.DataFrame({
        "timestamp": t_gps,
        "lat": lat,
        "lon": lon,
        "alt": alt,
        "speed": speed,
        "num_sats": rng.integers(8, 14, n_gps),
        "status": 3,
    })

    # ----- IMU track -----
    n_imu = int(duration_s * imu_hz)
    t_imu = np.linspace(0, duration_s, n_imu)

    # Gravity ~9.81 on Z, motion on X/Y
    accX = rng.normal(0.0, 0.3, n_imu)
    accY = rng.normal(0.0, 0.3, n_imu)
    # Vertical: lift during takeoff, descent, hover around 0 in cruise
    vert_profile = np.concatenate([
        np.linspace(0.5, 0.0, n_imu // 4),   # takeoff thrust
        np.zeros(n_imu // 2),
        np.linspace(0.0, -0.5, n_imu - 3 * (n_imu // 4)),  # descent
    ])
    accZ = vert_profile[:n_imu] + rng.normal(0, 0.1, n_imu) - 9.81

    imu_df = pd.DataFrame({
        "timestamp": t_imu,
        "accX": accX,
        "accY": accY,
        "accZ": accZ,
        "gyrX": rng.normal(0, 0.05, n_imu),
        "gyrY": rng.normal(0, 0.05, n_imu),
        "gyrZ": rng.normal(0, 0.02, n_imu),
    })

    return gps_df, imu_df
