from parser import parse_bin
from CoreAnalytics.coreAnalytics import merge_data, compute_all_metrics

for file in ["00000001.BIN", "00000019.BIN"]:
    print(f"\n--- {file} ---")
    
    gps_df, imu_df = parse_bin(file)
    df = merge_data(gps_df, imu_df)
    result = compute_all_metrics(df)

    for key, value in result.items():
        print(f"{key}: {value:.2f}")
