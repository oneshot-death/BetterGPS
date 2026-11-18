
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from dataset_loader import load_trip

TRAIN_DIR = BASE_DIR / "dataset" / "train"
TEST_DIR = BASE_DIR / "dataset" / "student_test"

OUTPUT_DIR = BASE_DIR / "outputs" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def analyse_network(folder, partition):

    track, nodes, arcs, route = load_trip(folder)

    n = len(nodes)

    if n == 0:
        raise ValueError("Network has no nodes")

    sources = arcs["from_node_id"].to_numpy(dtype=int)
    targets = arcs["to_node_id"].to_numpy(dtype=int)

    valid = (
        (sources >= 0) & (sources < n) &
        (targets >= 0) & (targets < n)
    )

    sources = sources[valid]
    targets = targets[valid]

    # Count node degrees
    indegree = np.bincount(targets, minlength=n)
    outdegree = np.bincount(sources, minlength=n)

    isolated_nodes = int(
        np.sum((indegree == 0) & (outdegree == 0))
    )

    # Duplicate directed arcs
    edge_array = np.column_stack((sources, targets))
    unique_edges = np.unique(edge_array, axis=0)
    duplicate_arcs = len(edge_array) - len(unique_edges)

    # Self-loops
    self_loops = int(np.sum(sources == targets))

    # Construct sparse directed graph
    graph = csr_matrix(
        (
            np.ones(len(sources), dtype=np.int8),
            (sources, targets)
        ),
        shape=(n, n)
    )

    # Weakly connected components
    component_count, labels = connected_components(
        graph,
        directed=True,
        connection="weak"
    )

    component_sizes = np.bincount(labels)

    largest_component = (
        int(component_sizes.max())
        if len(component_sizes) else 0
    )

    largest_component_percentage = (
        largest_component / n * 100
    )

    return {
        "partition": partition,
        "trip_id": folder.name,
        "total_nodes": n,
        "total_arcs": len(arcs),
        "valid_arcs": len(sources),
        "isolated_nodes": isolated_nodes,
        "duplicate_directed_arcs": duplicate_arcs,
        "self_loops": self_loops,
        "weakly_connected_components": int(component_count),
        "largest_component_nodes": largest_component,
        "largest_component_percentage": round(
            largest_component_percentage, 2
        ),
        "maximum_indegree": int(indegree.max()),
        "maximum_outdegree": int(outdegree.max())
    }


def main():

    results = []

    for partition, directory in [
        ("train", TRAIN_DIR),
        ("student_test", TEST_DIR)
    ]:

        print(f"\nAnalysing {partition} road networks...")

        folders = sorted(
            p for p in directory.iterdir() if p.is_dir()
        )

        for folder in folders:

            try:
                result = analyse_network(folder, partition)
                results.append(result)

                print(
                    f"Completed {folder.name} | "
                    f"Nodes: {result['total_nodes']} | "
                    f"Components: "
                    f"{result['weakly_connected_components']}"
                )

            except Exception as error:
                print(f"Error in {folder.name}: {error}")

    df = pd.DataFrame(results)

    output_file = OUTPUT_DIR / "road_network_statistics.csv"
    df.to_csv(output_file, index=False)

    print("\n========== ROAD NETWORK SUMMARY ==========")
    print("Successfully analysed:", len(df), "trips")

    print(
        df.groupby("partition")[
            [
                "total_nodes",
                "total_arcs",
                "isolated_nodes",
                "weakly_connected_components"
            ]
        ].agg(["mean", "min", "max"]).round(2)
    )

    print("\nSaved:", output_file)


if __name__ == "__main__":
    main()
