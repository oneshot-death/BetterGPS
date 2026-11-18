
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(BASE_DIR / "src"))

from dataset_loader import load_trip

TRAIN_DIR = BASE_DIR / "dataset" / "train"
TEST_DIR = BASE_DIR / "dataset" / "student_test"

OUTPUT_DIR = BASE_DIR / "outputs" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def analyse_trip(folder, partition):

    track, nodes, arcs, route = load_trip(folder)

    # GPS timestamp analysis
    timestamps = track["timestamp"]
    time_diff = timestamps.diff()

    gaps = time_diff[time_diff > 1]

    # GPS coordinate validation
    valid_coordinates = (
        track["longitude"].between(-180, 180)
        & track["latitude"].between(-90, 90)
        & np.isfinite(track["longitude"])
        & np.isfinite(track["latitude"])
    )

    # Road network validation
    valid_arcs = (
        arcs["from_node_id"].between(0, len(nodes) - 1)
        & arcs["to_node_id"].between(0, len(nodes) - 1)
    )

    # Reference route validation
    valid_route = (
        route["arc_id"].between(0, len(arcs) - 1)
    )

    return {
        "partition": partition,
        "trip_id": folder.name,

        "gps_points": len(track),
        "road_nodes": len(nodes),
        "road_arcs": len(arcs),
        "route_length": len(route),

        "missing_gps_values": int(
            track.isna().sum().sum()
        ),

        "invalid_coordinates": int(
            (~valid_coordinates).sum()
        ),

        "normal_1sec_intervals": int(
            (time_diff == 1).sum()
        ),

        "gps_gap_count": int(
            (time_diff > 1).sum()
        ),

        "duplicate_timestamps": int(
            (time_diff == 0).sum()
        ),

        "negative_time_intervals": int(
            (time_diff < 0).sum()
        ),

        "maximum_gap_seconds": (
            gaps.max() if len(gaps) else 0
        ),

        "invalid_arc_endpoints": int(
            (~valid_arcs).sum()
        ),

        "invalid_route_ids": int(
            (~valid_route).sum()
        )
    }


def main():

    results = []

    partitions = [
        ("train", TRAIN_DIR),
        ("student_test", TEST_DIR)
    ]

    for partition, directory in partitions:

        print(f"\nAnalysing {partition}...")

        folders = sorted(
            p for p in directory.iterdir()
            if p.is_dir()
        )

        for folder in folders:

            try:
                result = analyse_trip(folder, partition)
                results.append(result)

                print(
                    f"Completed trip {folder.name}"
                )

            except Exception as error:

                print(
                    f"Error in trip {folder.name}: {error}"
                )

    df = pd.DataFrame(results)

    # Save complete trip-wise statistics
    df.to_csv(
        OUTPUT_DIR / "trip_statistics.csv",
        index=False
    )

    # Generate partition-level summary
    numeric_columns = [
        "gps_points",
        "road_nodes",
        "road_arcs",
        "route_length",
        "missing_gps_values",
        "invalid_coordinates",
        "gps_gap_count",
        "invalid_arc_endpoints",
        "invalid_route_ids"
    ]

    summary = df.groupby("partition")[
        numeric_columns
    ].agg(["count", "sum", "mean", "min", "max"])

    summary.to_csv(
        OUTPUT_DIR / "dataset_summary.csv"
    )

    # Display final results
    print("\n========== EDA SUMMARY ==========")

    print("Successfully analysed trips:", len(df))

    print("\nTrip statistics:")
    print(df.head(10).to_string(index=False))

    print("\nPartition summary:")
    print(summary)

    print("\nCSV files saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
