
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
TABLE_DIR = BASE_DIR / "outputs" / "tables"

gaps = pd.read_csv(TABLE_DIR / "gps_gap_details.csv")
trips = pd.read_csv(TABLE_DIR / "gps_gap_summary.csv")

# 1. Partition-level statistics
partition_summary = trips.groupby("partition").agg(
    total_trips=("trip_id", "count"),
    trips_with_gaps=("gap_count", lambda x: (x > 0).sum()),
    total_gaps=("gap_count", "sum"),
    total_missing_seconds=("total_missing_seconds", "sum"),
    average_gaps_per_trip=("gap_count", "mean"),
    maximum_gap_in_any_trip=("maximum_gap_seconds", "max")
).reset_index()

print("\n========== PARTITION SUMMARY ==========")
print(partition_summary.to_string(index=False))

# 2. Ten trips with the most GPS gaps
top_gap_trips = trips.sort_values(
    "gap_count", ascending=False
).head(10)

print("\n========== TOP 10 TRIPS BY GAP COUNT ==========")
print(top_gap_trips.to_string(index=False))

# 3. Ten largest individual GPS gaps
largest_gaps = gaps.sort_values(
    "gap_seconds", ascending=False
).head(10)

print("\n========== TOP 10 LARGEST GPS GAPS ==========")
print(largest_gaps.to_string(index=False))

# 4. Save results
partition_summary.to_csv(
    TABLE_DIR / "gps_gap_partition_summary.csv", index=False
)

top_gap_trips.to_csv(
    TABLE_DIR / "top_10_gap_trips.csv", index=False
)

largest_gaps.to_csv(
    TABLE_DIR / "top_10_largest_gaps.csv", index=False
)

print("\nReports saved successfully!")
