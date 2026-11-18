
from pathlib import Path
import pandas as pd

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "dataset" / "train"
TEST_DIR = BASE_DIR / "dataset" / "student_test"

def load_trip(trip_folder):

    trip_id = trip_folder.name

    track = pd.read_csv(
        trip_folder / f"{trip_id}.track",
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude", "timestamp"]
    )

    nodes = pd.read_csv(
        trip_folder / f"{trip_id}.nodes",
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude"]
    )

    arcs = pd.read_csv(
        trip_folder / f"{trip_id}.arcs",
        sep=r"\s+",
        header=None,
        names=["from_node_id", "to_node_id"]
    )

    route = pd.read_csv(
        trip_folder / f"{trip_id}.route",
        header=None,
        names=["arc_id"]
    )

    return track, nodes, arcs, route


def main():

    for partition, directory in [
        ("TRAIN", TRAIN_DIR),
        ("STUDENT_TEST", TEST_DIR)
    ]:

        print(f"\n========== {partition} ==========")

        trips = sorted(
            p for p in directory.iterdir()
            if p.is_dir()
        )

        print("Number of trips:", len(trips))

        for folder in trips:

            track, nodes, arcs, route = load_trip(folder)

            print(
                folder.name,
                "| GPS:", len(track),
                "| Nodes:", len(nodes),
                "| Arcs:", len(arcs),
                "| Route:", len(route)
            )


if __name__ == "__main__":
    main()
