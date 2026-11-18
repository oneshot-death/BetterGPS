
from pathlib import Path
import sys
import folium

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from dataset_loader import load_trip

TRAIN_DIR = BASE_DIR / "dataset" / "train"
OUTPUT_DIR = BASE_DIR / "outputs" / "maps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Four representative trips
TRIP_IDS = ["00000090", "00000078", "00000098", "00000001"]

for TRIP_ID in TRIP_IDS:

    print(f"\nCreating map for trip {TRIP_ID}...")

    TRIP_DIR = TRAIN_DIR / TRIP_ID
    track, nodes, arcs, route = load_trip(TRIP_DIR)

    time_diff = track["timestamp"].diff()
    gap_idx = time_diff.idxmax()

    center_lat = track["latitude"].mean()
    center_lon = track["longitude"].mean()

    # Create map with Esri basemap
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles=None
    )

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri",
        name="Esri Street Map"
    ).add_to(m)

    # GPS trajectory
    gps_points = list(zip(track["latitude"], track["longitude"]))

    folium.PolyLine(
        gps_points,
        color="blue",
        weight=3,
        opacity=0.7,
        tooltip="GPS trajectory"
    ).add_to(m)

    # GPS markers
    for i in range(0, len(track), max(1, len(track) // 100)):
        folium.CircleMarker(
            location=[
                track.loc[i, "latitude"],
                track.loc[i, "longitude"]
            ],
            radius=2,
            color="blue",
            fill=True,
            popup=f"GPS point {i}"
        ).add_to(m)

    # Largest GPS gap
    before = track.loc[gap_idx - 1]
    after = track.loc[gap_idx]

    folium.Marker(
        [before["latitude"], before["longitude"]],
        tooltip="Before largest GPS gap",
        icon=folium.Icon(color="red")
    ).add_to(m)

    folium.Marker(
        [after["latitude"], after["longitude"]],
        tooltip="After largest GPS gap",
        icon=folium.Icon(color="red")
    ).add_to(m)

    folium.PolyLine(
        [
            [before["latitude"], before["longitude"]],
            [after["latitude"], after["longitude"]]
        ],
        color="red",
        weight=4,
        dash_array="8, 8",
        tooltip=f"Largest gap: {time_diff.loc[gap_idx]} seconds"
    ).add_to(m)

    # Reference route
    route_coordinates = []

    for arc_id in route["arc_id"].astype(int):
        arc = arcs.iloc[arc_id]

        start_node = nodes.iloc[int(arc["from_node_id"])]
        end_node = nodes.iloc[int(arc["to_node_id"])]

        route_coordinates.append(
            [start_node["latitude"], start_node["longitude"]]
        )
        route_coordinates.append(
            [end_node["latitude"], end_node["longitude"]]
        )

    folium.PolyLine(
        route_coordinates,
        color="green",
        weight=4,
        opacity=0.8,
        tooltip="Ground-truth reference route"
    ).add_to(m)

    # Nearby road network
    padding = 0.01

    nearby_nodes = (
        nodes["longitude"].between(
            track["longitude"].min() - padding,
            track["longitude"].max() + padding
        )
        & nodes["latitude"].between(
            track["latitude"].min() - padding,
            track["latitude"].max() + padding
        )
    )

    nearby_node_ids = set(nodes.index[nearby_nodes])

    nearby_arcs = arcs[
        arcs["from_node_id"].isin(nearby_node_ids)
        & arcs["to_node_id"].isin(nearby_node_ids)
    ]

    road_layer = folium.FeatureGroup(name="Nearby road network")

    for _, arc in nearby_arcs.iterrows():
        start = nodes.iloc[int(arc["from_node_id"])]
        end = nodes.iloc[int(arc["to_node_id"])]

        folium.PolyLine(
            [
                [start["latitude"], start["longitude"]],
                [end["latitude"], end["longitude"]]
            ],
            color="gray",
            weight=1,
            opacity=0.4
        ).add_to(road_layer)

    road_layer.add_to(m)
    folium.LayerControl().add_to(m)

    # Save map
    output_file = OUTPUT_DIR / f"trip_{TRIP_ID}_map.html"
    m.save(output_file)

    print("GPS points:", len(track))
    print("Largest gap:", time_diff.loc[gap_idx], "seconds")
    print("Nearby road arcs:", len(nearby_arcs))
    print("Reference route arcs:", len(route))
    print("Saved:", output_file)

print("\nAll four maps generated successfully!")
