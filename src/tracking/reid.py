from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import yaml

try:
    try:
        from torchreid.utils import FeatureExtractor
    except ModuleNotFoundError as exc:
        if exc.name != "torchreid.utils":
            raise
        from torchreid.reid.utils import FeatureExtractor
except ImportError as exc:
    FeatureExtractor = None
    _TORCHREID_IMPORT_ERROR = exc
else:
    _TORCHREID_IMPORT_ERROR = None


@dataclass
class IdentityState:
    global_id: int
    embedding: np.ndarray
    camera_id: str
    last_seen_seconds: float
    last_position: Tuple[float, float]
    local_tracks: Dict[str, int] = field(default_factory=dict)


class ClothesColorEmbedder:
    """Appearance embedding from upper-body color, not face features."""

    def __init__(self, hist_bins: Tuple[int, int] = (16, 16)) -> None:
        self.hist_bins = hist_bins

    def extract(self, frame: np.ndarray, box: Tuple[float, float, float, float]) -> np.ndarray:
        x1, y1, x2, y2 = map(int, box)
        height, width = frame.shape[:2]

        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            return np.zeros(self.hist_bins[0] * self.hist_bins[1], dtype=np.float32)

        person_crop = frame[y1:y2, x1:x2]
        upper_body = person_crop[: max(1, int(person_crop.shape[0] * 0.6)), :]

        hsv = cv2.cvtColor(upper_body, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv],
            [0, 1],
            None,
            self.hist_bins,
            [0, 180, 0, 256],
        )
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
        return hist


class OSNetEmbedder:
    """OSNet body ReID embedding via Torchreid; does not use face recognition."""

    def __init__(
        self,
        model_name: str = "osnet_x1_0",
        model_path: str = "",
        device: str = "cpu",
    ) -> None:
        if FeatureExtractor is None:
            raise RuntimeError(
                "Missing optional dependency: torchreid. Install torch + torchreid "
                "or run with `--reid-backend color`."
            ) from _TORCHREID_IMPORT_ERROR

        self.extractor = FeatureExtractor(
            model_name=model_name,
            model_path=model_path,
            device=device,
        )

    def extract(self, frame: np.ndarray, box: Tuple[float, float, float, float]) -> np.ndarray:
        x1, y1, x2, y2 = map(int, box)
        height, width = frame.shape[:2]

        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            return np.zeros(512, dtype=np.float32)

        crop = frame[y1:y2, x1:x2]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        features = self.extractor([crop])
        embedding = features[0].detach().cpu().numpy().astype(np.float32)
        return normalize_vector(embedding)


class CrossCameraReIdentifier:
    def __init__(
        self,
        similarity_threshold: float = 0.65,
        transition_window_seconds: float = 20.0,
        smoothing: float = 0.8,
        camera_config_path: Optional[str] = None,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.transition_window_seconds = transition_window_seconds
        self.smoothing = smoothing
        self.camera_transitions = load_camera_transitions(camera_config_path)
        self._next_global_id = 1
        self._identities: Dict[int, IdentityState] = {}

    def assign(
        self,
        embedding: np.ndarray,
        camera_id: str,
        local_track_id: int,
        timestamp_seconds: float,
        position: Tuple[float, float],
    ) -> int:
        existing = self._find_existing_local_identity(camera_id, local_track_id)
        if existing is not None:
            self._update_identity(
                existing,
                embedding,
                camera_id,
                local_track_id,
                timestamp_seconds,
                position,
            )
            return existing.global_id

        match = self._match_by_appearance(camera_id, embedding, timestamp_seconds, position)
        if match is not None:
            self._update_identity(
                match,
                embedding,
                camera_id,
                local_track_id,
                timestamp_seconds,
                position,
            )
            return match.global_id

        global_id = self._next_global_id
        self._next_global_id += 1
        self._identities[global_id] = IdentityState(
            global_id=global_id,
            embedding=embedding,
            camera_id=camera_id,
            last_seen_seconds=timestamp_seconds,
            last_position=position,
            local_tracks={camera_id: local_track_id},
        )
        return global_id

    def _find_existing_local_identity(
        self,
        camera_id: str,
        local_track_id: int,
    ) -> Optional[IdentityState]:
        for identity in self._identities.values():
            if identity.local_tracks.get(camera_id) == local_track_id:
                return identity
        return None

    def _match_by_appearance(
        self,
        camera_id: str,
        embedding: np.ndarray,
        timestamp_seconds: float,
        position: Tuple[float, float],
    ) -> Optional[IdentityState]:
        best_identity = None
        best_score = -1.0

        for identity in self._identities.values():
            elapsed = timestamp_seconds - identity.last_seen_seconds
            max_seconds = self._transition_window(identity.camera_id, camera_id)
            if elapsed < 0 or elapsed > max_seconds:
                continue

            if identity.camera_id == camera_id:
                continue

            if not self._camera_transition_allowed(identity.camera_id, camera_id):
                continue

            score = cosine_similarity(identity.embedding, embedding)
            score += transition_position_bonus(
                self.camera_transitions,
                from_camera=identity.camera_id,
                to_camera=camera_id,
                last_position=identity.last_position,
                current_position=position,
            )
            if score > best_score:
                best_score = score
                best_identity = identity

        if best_score >= self.similarity_threshold:
            return best_identity

        return None

    def _update_identity(
        self,
        identity: IdentityState,
        embedding: np.ndarray,
        camera_id: str,
        local_track_id: int,
        timestamp_seconds: float,
        position: Tuple[float, float],
    ) -> None:
        identity.embedding = (
            self.smoothing * identity.embedding
            + (1.0 - self.smoothing) * embedding
        )
        identity.camera_id = camera_id
        identity.last_seen_seconds = timestamp_seconds
        identity.last_position = position
        identity.local_tracks[camera_id] = local_track_id

    def _transition_window(self, from_camera: str, to_camera: str) -> float:
        transition = self.camera_transitions.get((from_camera, to_camera))
        if transition:
            return float(transition.get("max_seconds", self.transition_window_seconds))
        return self.transition_window_seconds

    def _camera_transition_allowed(self, from_camera: str, to_camera: str) -> bool:
        if not self.camera_transitions:
            return True
        return (from_camera, to_camera) in self.camera_transitions


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / denominator)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return vector
    return vector / norm


def load_camera_transitions(config_path: Optional[str]) -> Dict[Tuple[str, str], Dict]:
    if not config_path:
        return {}

    path = Path(config_path)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    transitions = {}
    for from_camera, camera_data in data.get("cameras", {}).items():
        for transition in camera_data.get("exits", []):
            to_camera = str(transition["to"])
            transitions[(str(from_camera), to_camera)] = transition

    return transitions


def transition_position_bonus(
    transitions: Dict[Tuple[str, str], Dict],
    from_camera: str,
    to_camera: str,
    last_position: Tuple[float, float],
    current_position: Tuple[float, float],
) -> float:
    transition = transitions.get((from_camera, to_camera))
    if not transition:
        return 0.0

    bonus = 0.0
    if point_in_normalized_box(last_position, transition.get("exit_zone")):
        bonus += 0.05
    if point_in_normalized_box(current_position, transition.get("entry_zone")):
        bonus += 0.05
    return bonus


def point_in_normalized_box(
    point: Tuple[float, float],
    zone: Optional[Tuple[float, float, float, float]],
) -> bool:
    if not zone:
        return False

    x1, y1, x2, y2 = zone
    x, y = point
    return x1 <= x <= x2 and y1 <= y <= y2
