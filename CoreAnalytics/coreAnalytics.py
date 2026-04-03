import numpy as np
import pandas as pd


def merge_data(gps_df, imu_df):
    df = pd.merge_asof(
        gps_df.sort_values("timestamp"),
        imu_df.sort_values("timestamp"),
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
        "accZ": "az_m_s2"
    })

    return df


def validate_columns(df):
    required = ["time_s", "lat_deg", "lon_deg", "alt_m",
                "ax_m_s2", "ay_m_s2", "az_m_s2"]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")


def prepare_data(df):
    df = df.sort_values("time_s").reset_index(drop=True)
    df["dt"] = df["time_s"].diff().fillna(0)
    return df


def compute_flight_duration(df):
    return df["time_s"].iloc[-1] - df["time_s"].iloc[0]


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000

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
    total = 0
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
    a = np.sqrt(
        df["ax_m_s2"] ** 2 +
        df["ay_m_s2"] ** 2 +
        df["az_m_s2"] ** 2
    )
    return a.max()


def integrate_velocity(df):
    df = df.copy()

    vx, vy, vz = [0.0], [0.0], [0.0]

    for i in range(1, len(df)):
        dt = df.loc[i, "dt"]

        vx.append(vx[-1] + (df.loc[i - 1, "ax_m_s2"] + df.loc[i, "ax_m_s2"]) * 0.5 * dt)
        vy.append(vy[-1] + (df.loc[i - 1, "ay_m_s2"] + df.loc[i, "ay_m_s2"]) * 0.5 * dt)
        vz.append(vz[-1] + (df.loc[i - 1, "az_m_s2"] + df.loc[i, "az_m_s2"]) * 0.5 * dt)

    df["vx"] = vx
    df["vy"] = vy
    df["vz"] = vz

    return df


def compute_max_horizontal_speed(df):
    if "gps_speed" in df.columns and df["gps_speed"].notna().any():
        return df["gps_speed"].max()

    df = integrate_velocity(df)
    speed = np.sqrt(df["vx"] ** 2 + df["vy"] ** 2)
    return speed.max()


def compute_all_metrics(df):
    validate_columns(df)
    df = prepare_data(df)

    return {
        "duration": float(compute_flight_duration(df)),
        "distance": float(compute_total_distance(df)),
        "max_altitude_gain": float(compute_max_altitude_gain(df)),
        "max_vertical_speed": float(compute_max_vertical_speed(df)),
        "max_acceleration": float(compute_max_acceleration(df)),
        "max_horizontal_speed": float(compute_max_horizontal_speed(df)),
    }
