import numpy as np
import pandas as pd


def merge_data(gps_df, imu_df, att_df, mode_df, curr_df):
    """
    Merge GPS, IMU, ATT, MODE, CURR data into one time-synchronized DataFrame.
    GPS is used as the base timeline.
    """
    if gps_df.empty:
        raise ValueError("GPS DataFrame is empty. Cannot build analytics timeline.")

    df = gps_df.sort_values("timestamp").copy()

    if not imu_df.empty:
        df = pd.merge_asof(
            df,
            imu_df.sort_values("timestamp"),
            on="timestamp",
            direction="nearest"
        )

    if not att_df.empty:
        df = pd.merge_asof(
            df,
            att_df.sort_values("timestamp"),
            on="timestamp",
            direction="nearest"
        )

    if not mode_df.empty:
        df = pd.merge_asof(
            df,
            mode_df.sort_values("timestamp"),
            on="timestamp",
            direction="nearest"
        )

    if not curr_df.empty:
        df = pd.merge_asof(
            df,
            curr_df.sort_values("timestamp"),
            on="timestamp",
            direction="nearest"
        )

    df = df.rename(columns={
        "timestamp": "time_s",
        "lat": "lat_deg",
        "lon": "lon_deg",
        "alt": "alt_m",
        "speed": "gps_speed",
        "vz": "gps_vz",
        "accX": "ax_m_s2",
        "accY": "ay_m_s2",
        "accZ": "az_m_s2",
        "mode_name": "flight_mode"
    })

    return df


def validate_columns(df):
    required = ["time_s", "lat_deg", "lon_deg", "alt_m"]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")


def prepare_data(df):
    df = df.sort_values("time_s").reset_index(drop=True)
    df["dt"] = df["time_s"].diff().fillna(0.0)
    return df


def compute_flight_duration(df):
    return df["time_s"].iloc[-1] - df["time_s"].iloc[0]


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def compute_total_distance(df):
    total = 0.0

    for i in range(1, len(df)):
        total += haversine(
            df.loc[i - 1, "lat_deg"],
            df.loc[i - 1, "lon_deg"],
            df.loc[i, "lat_deg"],
            df.loc[i, "lon_deg"]
        )

    return total


def compute_max_altitude_gain(df):
    return df["alt_m"].max() - df["alt_m"].iloc[0]


def compute_max_vertical_speed(df):
    if "gps_vz" in df.columns and df["gps_vz"].notna().any():
        return df["gps_vz"].abs().max()

    v = df["alt_m"].diff() / df["dt"].replace(0, np.nan)
    return np.abs(v).max()


def compute_max_acceleration(df):
    required_acc = ["ax_m_s2", "ay_m_s2", "az_m_s2"]

    if all(col in df.columns for col in required_acc):
        a = np.sqrt(
            df["ax_m_s2"] ** 2 +
            df["ay_m_s2"] ** 2 +
            df["az_m_s2"] ** 2
        )
        return a.max()

    return np.nan


def integrate_velocity(df):
    df = df.copy()

    required_acc = ["ax_m_s2", "ay_m_s2", "az_m_s2"]
    if not all(col in df.columns for col in required_acc):
        df["vx"] = np.nan
        df["vy"] = np.nan
        df["vz"] = np.nan
        return df

    vx, vy, vz = [0.0], [0.0], [0.0]

    for i in range(1, len(df)):
        dt = df.loc[i, "dt"]

        ax_prev = df.loc[i - 1, "ax_m_s2"]
        ax_curr = df.loc[i, "ax_m_s2"]
        ay_prev = df.loc[i - 1, "ay_m_s2"]
        ay_curr = df.loc[i, "ay_m_s2"]
        az_prev = df.loc[i - 1, "az_m_s2"]
        az_curr = df.loc[i, "az_m_s2"]

        vx.append(vx[-1] + (ax_prev + ax_curr) * 0.5 * dt)
        vy.append(vy[-1] + (ay_prev + ay_curr) * 0.5 * dt)
        vz.append(vz[-1] + (az_prev + az_curr) * 0.5 * dt)

    df["vx"] = vx
    df["vy"] = vy
    df["vz"] = vz

    return df


def compute_max_horizontal_speed(df):
    if "gps_speed" in df.columns and df["gps_speed"].notna().any():
        return df["gps_speed"].max()

    df = integrate_velocity(df)

    if df["vx"].notna().any() and df["vy"].notna().any():
        speed = np.sqrt(df["vx"] ** 2 + df["vy"] ** 2)
        return speed.max()

    return np.nan


def compute_min_voltage(df):
    if "voltage" in df.columns and df["voltage"].notna().any():
        return df["voltage"].min()
    return np.nan


def compute_max_current(df):
    if "current" in df.columns and df["current"].notna().any():
        return df["current"].max()
    return np.nan


def compute_total_consumed(df):
    if "consumed" in df.columns and df["consumed"].notna().any():
        return df["consumed"].max()
    return np.nan


def compute_max_roll(df):
    if "roll" in df.columns and df["roll"].notna().any():
        return df["roll"].abs().max()
    return np.nan


def compute_max_pitch(df):
    if "pitch" in df.columns and df["pitch"].notna().any():
        return df["pitch"].abs().max()
    return np.nan


def compute_max_yaw(df):
    if "yaw" in df.columns and df["yaw"].notna().any():
        return df["yaw"].abs().max()
    return np.nan


def get_flight_modes(df):
    if "flight_mode" in df.columns and df["flight_mode"].notna().any():
        return df["flight_mode"].dropna().unique().tolist()
    return []


def compute_all_metrics(df):
    validate_columns(df)
    df = prepare_data(df)

    return {
        "duration_s": float(compute_flight_duration(df)),
        "distance_m": float(compute_total_distance(df)),
        "max_altitude_gain_m": float(compute_max_altitude_gain(df)),
        "max_vertical_speed_m_s": float(compute_max_vertical_speed(df)),
        "max_acceleration_m_s2": float(compute_max_acceleration(df))
            if not np.isnan(compute_max_acceleration(df)) else np.nan,
        "max_horizontal_speed_m_s": float(compute_max_horizontal_speed(df))
            if not np.isnan(compute_max_horizontal_speed(df)) else np.nan,
        "min_voltage_V": float(compute_min_voltage(df))
            if not np.isnan(compute_min_voltage(df)) else np.nan,
        "max_current_A": float(compute_max_current(df))
            if not np.isnan(compute_max_current(df)) else np.nan,
        "consumed_mAh": float(compute_total_consumed(df))
            if not np.isnan(compute_total_consumed(df)) else np.nan,
        "max_roll_deg": float(compute_max_roll(df))
            if not np.isnan(compute_max_roll(df)) else np.nan,
        "max_pitch_deg": float(compute_max_pitch(df))
            if not np.isnan(compute_max_pitch(df)) else np.nan,
        "max_yaw_deg": float(compute_max_yaw(df))
            if not np.isnan(compute_max_yaw(df)) else np.nan,
        "flight_modes": get_flight_modes(df),
    }