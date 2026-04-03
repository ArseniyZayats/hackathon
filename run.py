from parser import parse_bin
from CoreAnalytics.coreAnalytics import merge_data, compute_all_metrics
import math 

for file in ["00000001.BIN", "00000019.BIN"]:
    print(f"\n--- {file} ---")

    gps_df, imu_df, att_df, mode_df, curr_df = parse_bin(file)
    df = merge_data(gps_df, imu_df, att_df, mode_df, curr_df)
    result = compute_all_metrics(df)

    for key, value in result.items():  
        if isinstance(value, float):
            if math.isnan(value):
                print(f"{key}: no data")
            else:
                print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")