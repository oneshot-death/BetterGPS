
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "outputs" / "tables" / "trip_statistics.csv"
PLOT_DIR = BASE_DIR / "outputs" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

sns.set_theme(style="whitegrid")

plots = [
    ("gps_points", "GPS Points per Trip", "GPS points", "gps_points_per_trip.png"),
    ("road_nodes", "Road Network Nodes per Trip", "Number of nodes", "road_nodes_per_trip.png"),
    ("road_arcs", "Road Network Arcs per Trip", "Number of arcs", "road_arcs_per_trip.png"),
    ("route_length", "Reference Route Length per Trip", "Number of route arcs", "route_length_per_trip.png"),
    ("gps_gap_count", "GPS Gaps per Trip", "Number of gaps", "gps_gaps_per_trip.png"),
]

for column, title, ylabel, filename in plots:
    plt.figure(figsize=(11, 6))
    sns.histplot(data=df, x=column, hue="partition", bins=25, kde=False)
    plt.title(title)
    plt.xlabel(ylabel)
    plt.ylabel("Number of trips")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=300)
    plt.close()
    print("Saved:", filename)

print("\nAll visualisations saved in:", PLOT_DIR)
