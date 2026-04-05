"""
parser.py — Ardupilot DataFlash (.bin) log parser

Extracts GPS and IMU messages into pandas DataFrames.

Ardupilot DataFlash binary format:
  - Each record starts with 0xA3 0x95 header bytes
  - Followed by message type byte, then payload
  - Format definitions come from FMT/FMTU messages (type ids, field names, units)

We rely on pymavlink to handle low-level binary decoding; our job is to
normalize the extracted fields into consistent units and structures.
"""

import numpy as np
import pandas as pd
from pymavlink import mavutil


# ---------------------------------------------------------------------------
# Units reported by Ardupilot FMTU messages for the fields we care about.
# pymavlink parses FMT/FMTU automatically and exposes them via msg.get_type()
# and field access, but the numeric values are already scaled — we only need
# to know whether a scaling step is still required.
#
# In practice, modern Ardupilot firmware stores:
#   GPS.Lat / GPS.Lng  — decimal degrees  (float, NOT integer * 1e7)
#   GPS.Alt            — meters AGL       (float)
#   GPS.Spd            — m/s              (float)
#   IMU.AccX/Y/Z       — m/s²            (float, body-frame, gravity included)
#   IMU.GyrX/Y/Z       — rad/s           (float)
#   All timestamps      — microseconds    (integer)
# ---------------------------------------------------------------------------

GPS_TYPES  = ("GPS", "GPS2", "GPSB")
IMU_TYPES  = ("IMU", "IMU2")
ATT_TYPES  = ("ATT",)
MODE_TYPES = ("MODE",)
CURR_TYPES = ("CURR", "BAT", "CURR2")

# Ardupilot flight mode ID → human-readable name.
# Covers both ArduCopter and ArduPlane (overlapping IDs get copter priority).
ARDUPILOT_MODES: dict[int, str] = {
    # ArduCopter
    0: "STABILIZE", 1: "ACRO", 2: "ALT_HOLD", 3: "AUTO",
    4: "GUIDED", 5: "LOITER", 6: "RTL", 7: "CIRCLE",
    9: "LAND", 11: "DRIFT", 13: "SPORT", 14: "FLIP",
    15: "AUTOTUNE", 16: "POSHOLD", 17: "BRAKE", 18: "THROW",
    19: "AVOID_ADSB", 20: "GUIDED_NOGPS", 21: "SMART_RTL",
    22: "FLOWHOLD", 23: "FOLLOW", 24: "ZIGZAG", 25: "SYSTEMID",
    26: "HELI_AUTOROTATE", 27: "AUTO_RTL",
    # ArduPlane extras (where id doesn't clash with copter above)
    8: "AUTOTUNE_FW", 10: "AUTO_FW", 12: "LOITER_FW",
    17: "QSTABILIZE", 18: "QHOVER", 19: "QLOITER",
    20: "QLAND", 21: "QRTL", 22: "QAUTOTUNE",
    23: "QACRO", 24: "THERMAL",
}


def parse_bin(filepath: str) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Parse an Ardupilot .bin DataFlash log file.

    Returns:
        gps_df  — timestamp(s), lat(°), lon(°), alt(m), speed(m/s), vz(m/s), num_sats, status
        imu_df  — timestamp(s), accX/Y/Z(m/s²), gyrX/Y/Z(rad/s)
        att_df  — timestamp(s), roll(°), pitch(°), yaw(°)
        mode_df — timestamp(s), mode_id, mode_name
        curr_df — timestamp(s), voltage(V), current(A), consumed(mAh)
    """
    all_types = [*GPS_TYPES, *IMU_TYPES, *ATT_TYPES, *MODE_TYPES, *CURR_TYPES]
    log = mavutil.mavlink_connection(filepath, dialect="ardupilotmega")

    gps_raw:  list[dict] = []
    imu_raw:  list[dict] = []
    att_raw:  list[dict] = []
    mode_raw: list[dict] = []
    curr_raw: list[dict] = []

    while True:
        msg = log.recv_match(type=all_types)
        if msg is None:
            break

        mtype = msg.get_type()

        if mtype in GPS_TYPES:
            lat = getattr(msg, "Lat", None)
            if lat is None:
                continue
            gps_raw.append({
                "timestamp": getattr(msg, "TimeUS", 0),
                "lat":       lat,
                "lon":       getattr(msg, "Lng", getattr(msg, "Lon", None)),
                "alt":       getattr(msg, "Alt", None),
                "speed":     getattr(msg, "Spd", None),
                "vz":        getattr(msg, "VZ", None),
                "num_sats":  getattr(msg, "NSats", None),
                "status":    getattr(msg, "Status", None),
            })

        elif mtype in IMU_TYPES:
            acc_x = getattr(msg, "AccX", None)
            if acc_x is None:
                continue
            imu_raw.append({
                "timestamp": getattr(msg, "TimeUS", 0),
                "accX": acc_x,
                "accY": getattr(msg, "AccY", None),
                "accZ": getattr(msg, "AccZ", None),
                "gyrX": getattr(msg, "GyrX", None),
                "gyrY": getattr(msg, "GyrY", None),
                "gyrZ": getattr(msg, "GyrZ", None),
            })

        elif mtype in ATT_TYPES:
            att_raw.append({
                "timestamp": getattr(msg, "TimeUS", 0),
                "roll":      getattr(msg, "Roll",  None),
                "pitch":     getattr(msg, "Pitch", None),
                "yaw":       getattr(msg, "Yaw",   None),
            })

        elif mtype in MODE_TYPES:
            mode_id = getattr(msg, "Mode", None)
            mode_raw.append({
                "timestamp": getattr(msg, "TimeUS", 0),
                "mode_id":   mode_id,
                "mode_name": ARDUPILOT_MODES.get(mode_id, f"MODE_{mode_id}"),
            })

        elif mtype in CURR_TYPES:
            volt = getattr(msg, "Volt", getattr(msg, "Voltage", None))
            if volt is None:
                continue
            curr_raw.append({
                "timestamp": getattr(msg, "TimeUS", 0),
                "voltage":   volt,
                "current":   getattr(msg, "Curr", getattr(msg, "Current", None)),
                "consumed":  getattr(msg, "CurrTot", getattr(msg, "Consumed", None)),
            })

    if not gps_raw and not imu_raw:
        raise ValueError(
            "No GPS or IMU messages found. "
            "File may be corrupted or not an Ardupilot DataFlash log."
        )

    gps_df  = _normalize_gps(pd.DataFrame(gps_raw))   if gps_raw  else pd.DataFrame()
    imu_df  = _normalize_imu(pd.DataFrame(imu_raw))   if imu_raw  else pd.DataFrame()
    att_df  = _normalize_ts(pd.DataFrame(att_raw))    if att_raw  else pd.DataFrame()
    mode_df = _normalize_ts(pd.DataFrame(mode_raw))   if mode_raw else pd.DataFrame()
    curr_df = _normalize_ts(pd.DataFrame(curr_raw))   if curr_raw else pd.DataFrame()

    return gps_df, imu_df, att_df, mode_df, curr_df


# ---------------------------------------------------------------------------
# Synthetic data generator — used for demo when no real .bin file is provided
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 2.4  Unit normalization
# ---------------------------------------------------------------------------

def _normalize_gps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw GPS fields to standard units.

    Modern Ardupilot firmware already outputs GPS.Lat/Lng as decimal degrees
    and Alt in metres. Older builds stored lat/lon as integer * 1e7 and alt
    in centimetres; we detect these cases heuristically only as a safety net.
    """
    df = df.copy()

    # TimeUS (µs) → seconds
    df["timestamp"] = df["timestamp"] / 1_000_000.0

    # Lat/Lon: if stored as integer * 1e7 the value exceeds ±360 substantially
    if df["lat"].abs().max() > 900:          # 900° is impossible in real degrees
        df["lat"] = df["lat"] / 1e7
        df["lon"] = df["lon"] / 1e7

    # Altitude: if stored in cm the value for typical flights exceeds 10 000
    if df["alt"].notna().any() and df["alt"].abs().max() > 10_000:
        df["alt"] = df["alt"] / 100.0

    df = df.dropna(subset=["lat", "lon"])
    # Filter out fixes with no reliable position (status < 3 = No Fix or 2D only)
    if "status" in df.columns:
        df = df[df["status"] >= 3]
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _normalize_imu(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw IMU fields to standard units.

    pymavlink already returns AccX/Y/Z in m/s² and GyrX/Y/Z in rad/s,
    so we only need to convert the timestamp.
    """
    df = df.copy()
    df["timestamp"] = df["timestamp"] / 1_000_000.0
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _normalize_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Convert TimeUS (µs) → seconds for ATT, MODE, CURR DataFrames."""
    df = df.copy()
    df["timestamp"] = df["timestamp"] / 1_000_000.0
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Sampling frequency utility
# ---------------------------------------------------------------------------

def sampling_frequencies(gps_df: pd.DataFrame, imu_df: pd.DataFrame) -> dict:
    """Return approximate sampling frequencies (Hz) for each sensor."""
    result: dict = {}

    if not gps_df.empty:
        dt = gps_df["timestamp"].diff().median()
        result["gps_hz"] = round(1.0 / dt, 2) if dt and dt > 0 else 0
        result["gps_count"] = len(gps_df)

    if not imu_df.empty:
        # Use mean of non-zero diffs to handle paired IMU/IMU2 entries
        diffs = imu_df["timestamp"].diff()
        dt = diffs[diffs > 0].mean()
        result["imu_hz"] = round(1.0 / dt, 2) if dt and dt > 0 else 0
        result["imu_count"] = len(imu_df)

    return result


# ---------------------------------------------------------------------------
# Synthetic data generator — used for demo when no real .bin file is provided
# ---------------------------------------------------------------------------

def generate_synthetic_flight(
    duration_s: float = 120.0,
    gps_hz: float = 5.0,
    imu_hz: float = 50.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    # Altitude: takeoff 0→50 m, cruise 50 m, descent 50→0 m
    q = n_gps // 4
    alt_profile = np.concatenate([
        np.linspace(0, 50, q),
        np.full(n_gps // 2, 50) + rng.normal(0, 0.5, n_gps // 2),
        np.linspace(50, 0, n_gps - 3 * q),
    ])
    alt = (alt_profile[:n_gps] + rng.normal(0, 0.2, n_gps)).clip(0)

    dx = np.gradient(lon, t_gps) * 111_320 * np.cos(np.radians(lat0))
    dy = np.gradient(lat, t_gps) * 110_540
    speed = np.hypot(dx, dy)
    vz = np.gradient(alt, t_gps)

    gps_df = pd.DataFrame({
        "timestamp": t_gps,
        "lat": lat,
        "lon": lon,
        "alt": alt,
        "speed": speed,
        "vz": vz,
        "num_sats": rng.integers(8, 14, n_gps),
        "status": 3,
    })

    # ----- IMU track -----
    n_imu = int(duration_s * imu_hz)
    t_imu = np.linspace(0, duration_s, n_imu)
    q_imu = n_imu // 4

    vert_profile = np.concatenate([
        np.linspace(0.5, 0.0, q_imu),
        np.zeros(n_imu // 2),
        np.linspace(0.0, -0.5, n_imu - 3 * q_imu),
    ])

    imu_df = pd.DataFrame({
        "timestamp": t_imu,
        "accX": rng.normal(0.0, 0.3, n_imu),
        "accY": rng.normal(0.0, 0.3, n_imu),
        "accZ": vert_profile[:n_imu] + rng.normal(0, 0.1, n_imu) - 9.81,
        "gyrX": rng.normal(0, 0.05, n_imu),
        "gyrY": rng.normal(0, 0.05, n_imu),
        "gyrZ": rng.normal(0, 0.02, n_imu),
    })

    # ----- ATT (attitude) -----
    att_df = pd.DataFrame({
        "timestamp": t_gps,
        "roll":  rng.normal(0, 5, n_gps),
        "pitch": rng.normal(0, 3, n_gps),
        "yaw":   np.degrees(phase) % 360,
    })

    # ----- MODE (flight modes) -----
    mode_ids = [0, 3, 5, 6]
    mode_df = pd.DataFrame({
        "timestamp": [0.0, duration_s * 0.1, duration_s * 0.2, duration_s * 0.85],
        "mode_id":   mode_ids,
        "mode_name": [ARDUPILOT_MODES.get(m, f"MODE_{m}") for m in mode_ids],
    })

    # ----- CURR (battery) -----
    n_curr = int(duration_s * 2)
    t_curr = np.linspace(0, duration_s, n_curr)
    curr_df = pd.DataFrame({
        "timestamp": t_curr,
        "voltage":   np.linspace(16.8, 14.2, n_curr) + rng.normal(0, 0.05, n_curr),
        "current":   np.abs(rng.normal(12, 2, n_curr)),
        "consumed":  np.linspace(0, 1800, n_curr),
    })

    return gps_df, imu_df, att_df, mode_df, curr_df