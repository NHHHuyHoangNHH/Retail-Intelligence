from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from tracking.reid import ClothesColorEmbedder, CrossCameraReIdentifier, OSNetEmbedder

try:
    from ultralytics import YOLO
except ImportError as exc:
    YOLO = None
    _YOLO_IMPORT_ERROR = exc
else:
    _YOLO_IMPORT_ERROR = None


PERSON_CLASS_ID = 0


@dataclass(frozen=True)
class PersonTrack:
    local_track_id: int
    global_id: int
    box: Tuple[float, float, float, float]
    confidence: float
    camera_id: str
    frame_id: int
    timestamp_seconds: float
    label: str = "person"

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class PersonTracker:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        tracker_config: str = "botsort.yaml",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        device: Optional[str] = None,
        image_size: int = 640,
        camera_id: str = "camera_1",
        reid_backend: str = "color",
        camera_config_path: Optional[str] = None,
        reidentifier: Optional[CrossCameraReIdentifier] = None,
    ) -> None:
        if YOLO is None:
            raise RuntimeError(
                "Missing dependency: ultralytics. Install it with "
                "`pip install -r requirements.txt`."
            ) from _YOLO_IMPORT_ERROR

        self.model = YOLO(model_path)
        self.tracker_config = tracker_config
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.image_size = image_size
        self.camera_id = camera_id
        self.embedder = build_embedder(reid_backend, device=device or "cpu")
        self.reidentifier = reidentifier or CrossCameraReIdentifier(
            camera_config_path=camera_config_path
        )

    def track(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp_seconds: float,
    ) -> List[PersonTrack]:
        if frame is None or frame.size == 0:
            return []

        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            classes=[PERSON_CLASS_ID],
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )

        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return []

        tracks: List[PersonTrack] = []

        for box in boxes:
            local_track_id = int(box.id[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            center = normalized_center((x1, y1, x2, y2), frame.shape)
            embedding = self.embedder.extract(frame, (x1, y1, x2, y2))
            global_id = self.reidentifier.assign(
                embedding=embedding,
                camera_id=self.camera_id,
                local_track_id=local_track_id,
                timestamp_seconds=timestamp_seconds,
                position=center,
            )

            tracks.append(
                PersonTrack(
                    local_track_id=local_track_id,
                    global_id=global_id,
                    box=(x1, y1, x2, y2),
                    confidence=confidence,
                    camera_id=self.camera_id,
                    frame_id=frame_id,
                    timestamp_seconds=timestamp_seconds,
                )
            )

        return tracks

    def track_video(
        self,
        video_path: str,
        frame_stride: int = 1,
        max_frames: Optional[int] = None,
    ) -> Iterable[Tuple[int, np.ndarray, List[PersonTrack]]]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_id = 0
        yielded = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_id % frame_stride == 0:
                    timestamp_seconds = frame_id / fps
                    tracks = self.track(frame, frame_id, timestamp_seconds)
                    yield frame_id, frame, tracks
                    yielded += 1

                    if max_frames is not None and yielded >= max_frames:
                        break

                frame_id += 1
        finally:
            cap.release()


def draw_person_tracks(
    frame: np.ndarray,
    tracks: List[PersonTrack],
    trail_history: Optional[Dict[int, Deque[Tuple[int, int]]]] = None,
    extra_labels: Optional[Dict[int, str]] = None,
) -> np.ndarray:
    output = frame.copy()
    extra_labels = extra_labels or {}

    for track in tracks:
        x1, y1, x2, y2 = map(int, track.box)
        color = color_for_id(track.global_id)
        extra_label = extra_labels.get(track.global_id)
        label = f"ID:{track.global_id}"
        if extra_label:
            label = f"{label} | {extra_label}"

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        if trail_history is not None:
            center = (int((x1 + x2) / 2), int(y2))
            trail_history[track.global_id].append(center)
            draw_trail(output, list(trail_history[track.global_id]), color)

        draw_label(output, label, (x1, y1), color)

    return output


def draw_label(
    frame: np.ndarray,
    label: str,
    origin: Tuple[int, int],
    color: Tuple[int, int, int],
) -> None:
    x, y = origin
    font_scale = 0.5
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )
    top = max(y - text_height - baseline - 6, 0)
    bottom = top + text_height + baseline + 6
    right = min(x + text_width + 8, frame.shape[1] - 1)

    cv2.rectangle(frame, (x, top), (right, bottom), color, -1)
    cv2.putText(
        frame,
        label,
        (x + 4, bottom - baseline - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def draw_trail(
    frame: np.ndarray,
    points: List[Tuple[int, int]],
    color: Tuple[int, int, int],
) -> None:
    if len(points) < 2:
        return

    for idx in range(1, len(points)):
        thickness = max(1, int(4 * idx / len(points)))
        cv2.line(frame, points[idx - 1], points[idx], color, thickness, cv2.LINE_AA)


def color_for_id(identity: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(identity)
    color = rng.integers(60, 255, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def build_embedder(reid_backend: str, device: str):
    if reid_backend == "osnet":
        return OSNetEmbedder(device=device)
    return ClothesColorEmbedder()


def normalized_center(
    box: Tuple[float, float, float, float],
    frame_shape: Tuple[int, int, int],
) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    height, width = frame_shape[:2]
    return ((x1 + x2) / (2.0 * width), (y1 + y2) / (2.0 * height))


def save_tracking_video(
    video_path: str,
    output_path: str,
    model_path: str = "yolov8n.pt",
    tracker_config: str = "botsort.yaml",
    camera_id: str = "camera_1",
    reid_backend: str = "color",
    camera_config_path: Optional[str] = None,
    trail_length: int = 45,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
) -> int:
    tracker = PersonTracker(
        model_path=model_path,
        tracker_config=tracker_config,
        camera_id=camera_id,
        reid_backend=reid_backend,
        camera_config_path=camera_config_path,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / max(frame_stride, 1),
        (width, height),
    )

    written = 0
    trail_history: Dict[int, Deque[Tuple[int, int]]] = defaultdict(
        lambda: deque(maxlen=trail_length)
    )
    try:
        for _, frame, tracks in tracker.track_video(
            video_path=video_path,
            frame_stride=frame_stride,
            max_frames=max_frames,
        ):
            writer.write(draw_person_tracks(frame, tracks, trail_history=trail_history))
            written += 1
    finally:
        writer.release()

    return written
