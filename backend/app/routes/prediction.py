
from fastapi import APIRouter, HTTPException

from app.schemas.request import PredictionRequest
from app.services.predictor import predict_route

router = APIRouter()


@router.post("/predict")
def predict(request: PredictionRequest):

    try:
        # Convert the incoming request into the format
        # expected by our ML pipeline.

        track = [
            [
                point.longitude,
                point.latitude,
                point.timestamp,
            ]
            for point in request.track
        ]

        nodes = [
            [
                node.longitude,
                node.latitude,
            ]
            for node in request.nodes
        ]

        arcs = [
            [
                arc.from_node,
                arc.to_node,
            ]
            for arc in request.arcs
        ]

        # Run the complete prediction pipeline.
        result = predict_route(
            track=track,
            nodes=nodes,
            arcs=arcs,
            trip_id=request.trip_id,
        )

        return {
            "status": "success",
            **result,
        }

    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}",
        ) from error
