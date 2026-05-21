from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from tracking.person_tracker import PersonTrack

try:
    from insightface.app import FaceAnalysis
except ImportError as exc:
    FaceAnalysis = None
    _INSIGHTFACE_IMPORT_ERROR = exc
else:
    _INSIGHTFACE_IMPORT_ERROR = None


@dataclass(frozen=True)
class DemographicObservation:
    global_id: int
    apparent_age_group: str
    gender: str
    confidence: float
    method: str


class HeuristicDemographicEstimator:
    def estimate(
        self,
        frame: np.ndarray,
        tracks: Sequence[PersonTrack],
    ) -> List[DemographicObservation]:
        return [estimate_body_scale_demographics(track, frame.shape) for track in tracks]


class BuffaloLDemographicEstimator:
    def __init__(
        self,
        model_root: Optional[str] = None,
        providers: Optional[List[str]] = None,
    ) -> None:
        if FaceAnalysis is None:
            raise RuntimeError(
                "Missing optional dependency: insightface. Install insightface "
                "and onnxruntime to use --demographics-backend buffalo_l."
            ) from _INSIGHTFACE_IMPORT_ERROR

        kwargs = {"name": "buffalo_l"}
        if model_root:
            kwargs["root"] = str(Path(model_root))

        self.app = FaceAnalysis(
            **kwargs,
            providers=providers or ["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def estimate(
        self,
        frame: np.ndarray,
        tracks: Sequence[PersonTrack],
    ) -> List[DemographicObservation]:
        faces = self.app.get(frame)
        observations = []

        for track in tracks:
            face = best_face_for_track(faces, track.box)
            if face is None:
                observations.append(
                    DemographicObservation(
                        global_id=track.global_id,
                        apparent_age_group="unknown",
                        gender="unknown",
                        confidence=0.0,
                        method="buffalo_l_no_face_match",
                    )
                )
                continue

            age = int(getattr(face, "age", -1))
            gender_value = int(getattr(face, "gender", -1))
            observations.append(
                DemographicObservation(
                    global_id=track.global_id,
                    apparent_age_group=age_to_group(age),
                    gender=gender_to_label(gender_value),
                    confidence=0.85,
                    method="buffalo_l",
                )
            )

        return observations


def build_demographic_estimator(
    backend: str,
    model_root: Optional[str] = None,
):
    if backend == "none":
        return None
    if backend == "buffalo_l":
        return BuffaloLDemographicEstimator(model_root=model_root)
    return HeuristicDemographicEstimator()


def estimate_body_scale_demographics(
    track: PersonTrack,
    frame_shape: Tuple[int, int, int],
) -> DemographicObservation:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = track.box
    box_height_ratio = max(y2 - y1, 0.0) / max(height, 1)
    box_width_ratio = max(x2 - x1, 0.0) / max(width, 1)
    touches_border = x1 <= 2 or y1 <= 2 or x2 >= width - 2 or y2 >= height - 2

    if touches_border or box_height_ratio < 0.12:
        age_group = "unknown"
        confidence = 0.25
    elif box_height_ratio < 0.34 and box_width_ratio < 0.18:
        age_group = "child_or_teen"
        confidence = 0.45
    else:
        age_group = "adult"
        confidence = 0.55

    return DemographicObservation(
        global_id=track.global_id,
        apparent_age_group=age_group,
        gender="unknown",
        confidence=confidence,
        method="body_scale_heuristic",
    )


def best_face_for_track(faces, box: Tuple[float, float, float, float]):
    x1, y1, x2, y2 = box
    candidates = []
    for face in faces:
        fx1, fy1, fx2, fy2 = map(float, face.bbox)
        cx = (fx1 + fx2) / 2.0
        cy = (fy1 + fy2) / 2.0
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            area = max(fx2 - fx1, 0.0) * max(fy2 - fy1, 0.0)
            candidates.append((area, face))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def age_to_group(age: int) -> str:
    if age < 0:
        return "unknown"
    if age < 13:
        return "child"
    if age < 20:
        return "teen"
    if age < 35:
        return "young_adult"
    if age < 60:
        return "adult"
    return "senior"


def gender_to_label(gender_value: int) -> str:
    mapping: Dict[int, str] = {
        0: "female",
        1: "male",
    }
    return mapping.get(gender_value, "unknown")
