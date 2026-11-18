from pydantic import BaseModel, Field, model_validator


class GPSPoint(BaseModel):
    longitude: float = Field(ge=-180, le=180)
    latitude: float = Field(ge=-90, le=90)
    timestamp: float


class RoadNode(BaseModel):
    longitude: float = Field(ge=-180, le=180)
    latitude: float = Field(ge=-90, le=90)


class RoadArc(BaseModel):
    from_node: int = Field(ge=0)
    to_node: int = Field(ge=0)


class PredictionRequest(BaseModel):
    trip_id: str = Field(min_length=1)
    track: list[GPSPoint] = Field(min_length=1)
    nodes: list[RoadNode] = Field(min_length=2)
    arcs: list[RoadArc] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_arc_nodes(self):
        node_count = len(self.nodes)

        for arc in self.arcs:
            if arc.from_node >= node_count or arc.to_node >= node_count:
                raise ValueError("Road arc references a node that does not exist")

        return self