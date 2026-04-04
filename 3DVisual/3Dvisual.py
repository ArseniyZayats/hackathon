"""
3Dvisual.py - Ardupilot UAV Telemetry 3D Visualizer

This script consumes parsed .BIN log DataFrames (GPS, IMU, Attitude, Modes, Power)
and generates an interactive, animated 3D HTML dashboard. 

Workflow:
1. Dynamically finds all .BIN files in the target directory using glob.
2. Uses the custom parser (parse_bin) to extract MAVLink data.
3. Merges asynchronous sensor data (e.g., 5Hz GPS with 50Hz IMU) using time-based proximity.
4. Converts spherical coordinates (Lat/Lon/Alt) to local Cartesian (East/North/Up).
5. Generates an interactive Plotly 3D scene with synchronized multi-flight animation.

Outputs:
- ultimate_analytics.html (Interactive dashboard saved in the current directory)
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os
import glob

# =============================================================================
# MODULE 1: PROJECT ENVIRONMENT & DEPENDENCIES
# =============================================================================
# Dynamically add the parent directory to sys.path so the script can locate 
# custom modules ('parser' and 'coreAnalytics') from any working directory.
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)

if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

# Import custom binary parser and data preparation tools
try:
    from parser import parse_bin
except ModuleNotFoundError:
    from Parsing.parser import parse_bin

try:
    from CoreAnalytics.coreAnalytics import merge_data, validate_columns, prepare_data
except ModuleNotFoundError:
    from coreAnalytics import merge_data, validate_columns, prepare_data

# =============================================================================
# MODULE 2: GEODETIC TO CARTESIAN TRANSFORMATION (ENU)
# =============================================================================
def geodetic2enu_custom(lat, lon, alt, lat0, lon0, alt0):
    """
    Converts WGS-84 spherical coordinates to local Cartesian (ENU) coordinates.
    Standard 3D rendering engines require flat grid coordinates (meters).
    
    Args:
        lat, lon, alt: Arrays of current target coordinates (Degrees, Meters)
        lat0, lon0, alt0: Reference "Home" coordinates (Take-off point)
        
    Returns:
        tuple[East, North, Up]: Arrays of distances in meters from the Home point.
    """
    # WGS-84 Ellipsoid constants
    a, b = 6378137.0, 6356752.314245
    e_sq = 1.0 - (b ** 2) / (a ** 2)

    def geodetic2ecef(la, lo, al):
        la_r, lo_r = np.radians(la), np.radians(lo)
        N = a / np.sqrt(1 - e_sq * np.sin(la_r)**2)
        return (N + al) * np.cos(la_r) * np.cos(lo_r), \
               (N + al) * np.cos(la_r) * np.sin(lo_r), \
               (N * (1 - e_sq) + al) * np.sin(la_r)

    # Convert both current point and Home point to Earth-Centered, Earth-Fixed
    x, y, z = geodetic2ecef(lat, lon, alt)
    x0, y0, z0 = geodetic2ecef(lat0, lon0, alt0)
    
    # Calculate difference and apply rotation matrix to get Local Tangent Plane
    dx, dy, dz = x - x0, y - y0, z - z0
    la0_r, lo0_r = np.radians(lat0), np.radians(lon0)
    sl, cl, sn, cn = np.sin(la0_r), np.cos(la0_r), np.sin(lo0_r), np.cos(lo0_r)
    
    east = -sn * dx + cn * dy
    north = -sl * cn * dx - sl * sn * dy + cl * dz
    up = cl * cn * dx + cl * sn * dy + sl * dz
    
    return east, north, up

# =============================================================================
# MODULE 3: MULTI-FILE PROCESSING & FEATURE ENGINEERING
# =============================================================================
# Automatically locate all .BIN files in the base directory
bin_files_path = os.path.join(base_dir, "*.BIN")
files_to_process = [os.path.basename(f) for f in glob.glob(bin_files_path)]

if not files_to_process:
    print(f"ERROR: No .BIN files found in {base_dir}")
    sys.exit()

processed_flights = []
global_speeds = []
max_flight_duration = 0

for bin_file_name in files_to_process:
    bin_file_path = os.path.join(base_dir, bin_file_name)
    
    try:
        # Extract raw dataframes using the parser
        parsed_data = parse_bin(bin_file_path)
        
        if not parsed_data or not isinstance(parsed_data, tuple):
            print(f"WARNING: Skipping {bin_file_name} - Corrupted format.")
            continue
            
        if len(parsed_data) == 5:
            gps_df, imu_df, att_df, mode_df, curr_df = parsed_data
        else:
            gps_df, imu_df = parsed_data[:2]
            att_df = mode_df = curr_df = pd.DataFrame()

        if gps_df.empty or imu_df.empty:
            print(f"WARNING: Skipping {bin_file_name} - No GPS or IMU data found.")
            continue

        # Time-based asynchronous merging (merge_asof) matches high-frequency IMU 
        # data to low-frequency GPS data based on the closest timestamp.
        real_df = merge_data(gps_df, imu_df)
        for extra_df in [att_df, mode_df, curr_df]:
            if not extra_df.empty:
                extra_df = extra_df.rename(columns={"timestamp": "time_s"})
                real_df = pd.merge_asof(real_df.sort_values("time_s"), 
                                        extra_df.sort_values("time_s"), 
                                        on="time_s", direction="nearest")
                
        validate_columns(real_df)
        df = prepare_data(real_df)
        
        required_cols = ['lat_deg', 'lon_deg', 'alt_m', 'time_s']
        if df.empty or not all(col in df.columns for col in required_cols):
            print(f"WARNING: Skipping {bin_file_name} - Missing core GPS data!")
            continue
            
        if 'gps_speed' not in df.columns:
            df['gps_speed'] = 0.0

        # Calculate derived telemetry features
        # 1. Vertical Speed (Climb rate)
        if 'v_speed' not in df.columns:
            df['v_speed'] = df['alt_m'].diff() / df['time_s'].diff().replace(0, np.nan)
            df['v_speed'] = df['v_speed'].fillna(0.0)

        # 2. ENU Cartesian Coordinates (Setting first point as Home [0,0,0])
        lat0, lon0, alt0 = df['lat_deg'].iloc[0], df['lon_deg'].iloc[0], df['alt_m'].iloc[0]
        df['E'], df['N'], df['U'] = geodetic2enu_custom(df['lat_deg'], df['lon_deg'], df['alt_m'], lat0, lon0, alt0)
        
        # 3. Mission Range (Euclidean distance from Home point)
        df['range'] = np.sqrt(df['E']**2 + df['N']**2 + df['U']**2)
        
        # 4. Relative Time (Crucial for multi-flight animation synchronization)
        df['t_rel'] = df['time_s'] - df['time_s'].iloc[0]
        duration = df['t_rel'].iloc[-1]
        if duration > max_flight_duration:
            max_flight_duration = duration
        
        global_speeds.extend(df['gps_speed'].tolist())
        processed_flights.append({"name": bin_file_name, "df": df, "alt_ref": alt0})

    except Exception as e:
        print(f"ERROR processing {bin_file_name}: {e}")
        continue

if not global_speeds:
    print("ERROR: No GPS speed data could be extracted from any files.")
    sys.exit()

# Establish global speed bounds for a unified color scale across all flights
min_s, max_s = min(global_speeds), max(global_speeds)

if min_s == max_s:
    max_s += 0.1

# =============================================================================
# MODULE 4: 3D VISUALIZATION ENGINE (STATIC BASE)
# =============================================================================
fig = go.Figure()

# Persistent dummy trace to render the speed colorbar regardless of active layers
fig.add_trace(go.Scatter3d(
    x=[None], y=[None], z=[None], mode='markers',
    marker=dict(colorscale='Turbo', cmin=min_s, cmax=max_s, showscale=True, 
                colorbar=dict(title=dict(text="Speed (m/s)", font=dict(size=14)), thickness=15, x=1.1)),
    showlegend=False
))

dashboard_text = "<b>UAV MISSION ANALYTICS</b><br><br>"

# Arrays to store Plotly trace indices. These are required later by the 
# animation module to know exactly which traces to update per frame.
main_line_indices = []
uav_marker_indices = []

for idx, flight in enumerate(processed_flights):
    df = flight["df"]
    short_id = flight['name'].replace('.BIN', '')[-2:]
    f_group = f"Flight: {flight['name']}"
    t_0 = df['time_s'].iloc[0]

    # Generate rich HTML hover tooltips with fallback logic for missing sensors
    hover_texts = []
    for _, row in df.iterrows():
        sat_txt = f"Sats: {int(row.get('nsat', 0))}" if 'nsat' in row else "Sats: N/A"
        pwr_val = row.get('volt', 0)
        pwr_txt = f"{pwr_val:.1f}V | {row.get('curr', 0):.1f}A" if pwr_val > 1.0 else "No Power Data"
        
        txt = (f"<b>ID: {short_id} [{row.get('mode_name', 'N/A')}]</b><br>"
               f"Time: {row.get('time_s', t_0) - t_0:.1f}s | {sat_txt}<br>"
               f"Speed: {row.get('gps_speed', 0.0):.1f} m/s | Climb: {row.get('v_speed', 0.0):+.1f} m/s<br>"
               f"Altitude: {row.get('alt_m', 0.0):.1f} m<br>"
               f"Attitude: R:{row.get('roll',0):.1f} P:{row.get('pitch',0):.1f} Y:{row.get('yaw',0):.1f} deg<br>"
               f"Power: {pwr_txt}")
        hover_texts.append(txt)

    # Store index for the main flight path trace (used by animation)
    main_line_indices.append(len(fig.data))
    
    # TRACE 1: Main 3D Flight Path
    fig.add_trace(go.Scatter3d(
        x=df['E'], y=df['N'], z=df['U'], mode='lines+markers',
        name=f"Flight {short_id}", legendgroup=f_group,
        text=hover_texts, hoverinfo="text",
        marker=dict(size=3, color=df['gps_speed'], colorscale='Turbo', cmin=min_s, cmax=max_s, line=dict(width=0)),
        line=dict(color=df['gps_speed'], colorscale='Turbo', width=4)
    ))

    # TRACE 2: Ground Shadow Projection (for depth awareness)
    fig.add_trace(go.Scatter3d(
        x=df['E'], y=df['N'], z=[0]*len(df), mode='lines',
        name="Ground Track", legendgroup=f_group, showlegend=False,
        line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dot'), hoverinfo="skip"
    ))

    # TRACE 3: Take-off and Landing Markers
    for i, lbl, col, sym in [(0, "START", "lime", "circle"), (-1, "END", "red", "diamond")]:
        row = df.iloc[i]
        fig.add_trace(go.Scatter3d(
            x=[row['E']], y=[row['N']], z=[row['U']], mode='markers+text',
            marker=dict(size=8, color=col, symbol=sym, line=dict(width=1, color='white')),
            showlegend=False, legendgroup=f_group,
            text=[f"<b>{lbl}</b>"], textposition="top center", textfont=dict(size=10, color="white"), hoverinfo="skip"
        ))

    # Compile global statistics for the on-screen dashboard
    dist = float(np.sum(np.sqrt(np.diff(df['E'])**2 + np.diff(df['N'])**2 + np.diff(df['U'])**2)))
    dashboard_text += f"<b>{flight['name']}</b><br>"
    dashboard_text += f"Distance: {dist:.0f}m | Avg Speed: {df['gps_speed'].mean():.1f}m/s<br>"
    dashboard_text += f"Max Alt: {df['alt_m'].max()-flight['alt_ref']:.1f}m | Max Range: {df['range'].max():.0f}m<br><br>"

# Add hidden "UAV Head" markers at the start of each trace. 
# These will move along the path during the animation playback.
for flight in processed_flights:
    df = flight["df"]
    short_id = flight['name'].replace('.BIN', '')[-2:]
    
    uav_marker_indices.append(len(fig.data))
    fig.add_trace(go.Scatter3d(
        x=[df['E'].iloc[0]], y=[df['N'].iloc[0]], z=[df['U'].iloc[0]],
        mode='markers', name=f"Moving UAV {short_id}", legendgroup=f"Flight: {flight['name']}",
        marker=dict(size=8, color=[df['gps_speed'].iloc[0]], colorscale='Turbo', cmin=min_s, cmax=max_s, line=dict(width=2, color='white')),
        hoverinfo="skip", showlegend=False
    ))

# =============================================================================
# MODULE 5: FLAWLESS ANIMATION LOGIC
# =============================================================================
# Generate equidistant time steps based on the longest flight to sync playback
NUM_FRAMES = 100
time_steps = np.linspace(0, max_flight_duration, NUM_FRAMES)
frames = []

for k, current_time in enumerate(time_steps):
    frame_data = []
    traces_to_update = []
    
    for i, flight in enumerate(processed_flights):
        df = flight["df"]
        
        # Locate the exact index in the dataframe that corresponds to the current animation time.
        # This allows for smooth drawing without downsampling the original high-res data.
        idx = np.searchsorted(df['t_rel'], current_time, side='right')
        if idx == 0: idx = 1 
        
        df_slice = df.iloc[:idx]
        
        # Draw the line up to the current progress point
        frame_data.append(go.Scatter3d(
            x=df_slice['E'], y=df_slice['N'], z=df_slice['U'],
            marker=dict(color=df_slice['gps_speed']),
            line=dict(color=df_slice['gps_speed'])
        ))
        traces_to_update.append(main_line_indices[i])
        
        # Update the position of the UAV marker head
        last_row = df_slice.iloc[-1]
        frame_data.append(go.Scatter3d(
            x=[last_row['E']], y=[last_row['N']], z=[last_row['U']],
            marker=dict(color=[last_row['gps_speed']])
        ))
        traces_to_update.append(uav_marker_indices[i])
        
    frames.append(go.Frame(data=frame_data, name=f"fr{k}", traces=traces_to_update))

fig.frames = frames

# =============================================================================
# FINAL UI LAYOUT
# =============================================================================
fig.add_annotation(
    text=dashboard_text, align='left', showarrow=False, xref='paper', yref='paper', x=0.01, y=0.01,
    bgcolor="rgba(15, 15, 15, 0.9)", bordercolor="white", borderwidth=1, borderpad=10, font=dict(color="white", size=10)
)

fig.update_layout(
    title="UAV Advanced Mission Analytics Dashboard",
    scene=dict(xaxis_title='East (m)', yaxis_title='North (m)', zaxis_title='Altitude (m)', aspectmode='data'),
    template="plotly_dark", 
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    margin=dict(l=0, r=0, b=0, t=40),
    hoverlabel=dict(bgcolor="rgba(20, 20, 20, 0.9)", font_size=12, font_color="white"),
    
    # Combined Play/Pause Control Button
    updatemenus=[dict(
        type="buttons", showactive=False, x=0.01, y=0.85, xanchor="left", yanchor="top",
        buttons=[dict(
            label="Play / Pause",
            method="animate",
            args=[None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}],
            args2=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]
        )]
    )],
    # Progress Slider Setup
    sliders=[dict(
        steps=[dict(args=[[f"fr{k}"], dict(frame=dict(duration=0, redraw=True), mode="immediate")], label=f"{k}%", method="animate") for k in range(NUM_FRAMES)],
        transition=dict(duration=0), x=0.01, xanchor="left", y=0, yanchor="top",
        currentvalue=dict(font=dict(size=12, color="white"), prefix="Progress: ", visible=True, xanchor="right")
    )]
)

output_file = os.path.join(current_dir, "ultimate_analytics.html")
# auto_play=False is critical to prevent the animation from starting on initial load
fig.write_html(output_file, auto_open=True, auto_play=False) 
print(f"Mission Successful! Analytics saved to: {output_file}")