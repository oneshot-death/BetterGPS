
from pathlib import Path
import sys
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from dataset_loader import load_trip

TRAIN_DIR = BASE_DIR / "dataset" / "train"
TEST_DIR = BASE_DIR / "dataset" / "student_test"

OUTPUT_DIR = BASE_DIR / "outputs" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

gap_records = []
summary_records = []

for partition, directory in [
    ("train", TRAIN_DIR),
    ("student_test", TEST_DIR)
]:
    print(f"\nAnalysing GPS gaps in {partition}...")

    for folder in sorted(p for p in directory.iterdir() if p.is_dir()):
        track, nodes, arcs, route = load_trip(folder)

        time_diff = track["timestamp"].diff()
        gap_indices = time_diff[time_diff > 1].index

        total_missing_seconds = 0

        for idx in gap_indices:
            gap_seconds = time_diff.loc[idx]
            missing_seconds = max(0, gap_seconds - 1)
            total_missing_seconds += missing_seconds

            gap_records.append({
                "partition": partition,
                "trip_id": folder.name,
                "previous_timestamp": track.loc[idx - 1, "timestamp"],
                "current_timestamp": track.loc[idx, "timestamp"],
                "gap_seconds": gap_seconds,
                "missing_seconds": missing_seconds,
                "previous_longitude": track.loc[idx - 1, "longitude"],
                "previous_latitude": track.loc[idx - 1, "latitude"],
                "current_longitude": track.loc[idx, "longitude"],
                "current_latitude": track.loc[idx, "latitude"]
            })

        summary_records.append({
            "partition": partition,
            "trip_id": folder.name,
            "gps_points": len(track),
            "gap_count": len(gap_indices),
            "total_missing_seconds": total_missing_seconds,
            "maximum_gap_seconds": (
                time_diff.loc[gap_indices].max()
                if len(gap_indices) else 0
            )
        })

        print(f"{folder.name}: {len(gap_indices)} gaps")

gap_df = pd.DataFrame(gap_records)
summary_df = pd.DataFrame(summary_records)

gap_df.to_csv(OUTPUT_DIR / "gps_gap_details.csv", index=False)
summary_df.to_csv(OUTPUT_DIR / "gps_gap_summary.csv", index=False)

print("\n========== GPS GAP ANALYSIS ==========")
print("Total GPS gaps:", len(gap_df))
print("Trips analysed:", len(summary_df))
print("\nPartition summary:")
print(
    summary_df.groupby("partition")[
        ["gps_points", "gap_count", "total_missing_seconds"]
    ].sum()
)

print("\nFiles saved in:", OUTPUT_DIR)
