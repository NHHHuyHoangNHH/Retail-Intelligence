from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from detection.types import Detection

try:
    from ultralytics import YOLO
except ImportError as exc:
    YOLO = None
    _YOLO_IMPORT_ERROR = exc
else:
    _YOLO_IMPORT_ERROR = None


PERSON_CLASS_ID = 0


class PersonDetector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.5,
        device: Optional[str] = None,
        image_size: int = 640,
    ) -> None:
        if YOLO is None:
            raise RuntimeError(
                "Missing dependency: ultralytics. Install it with "
                "`pip install ultralytics opencv-python`."
            ) from _YOLO_IMPORT_ERROR

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.image_size = image_size
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if frame is None or frame.size == 0:
            return []

        results = self.model.predict(
            source=frame,
            classes=[PERSON_CLASS_ID],
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )

        if not results:
            return []

        detections: List[Detection] = []
        boxes = results[0].boxes

        if boxes is None:
            return detections

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])

            detections.append(
                Detection(
                    box=(x1, y1, x2, y2),
                    confidence=confidence,
                    class_id=class_id,
                    label="person",
                )
            )

        return detections

    def detect_video(
        self,
        video_path: str,
        frame_stride: int = 1,
        max_frames: Optional[int] = None,
    ) -> Iterable[Tuple[int, np.ndarray, List[Detection]]]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        frame_id = 0
        yielded = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_id % frame_stride == 0:
                    yield frame_id, frame, self.detect(frame)
                    yielded += 1

                    if max_frames is not None and yielded >= max_frames:
                        break

                frame_id += 1
        finally:
            cap.release()


def draw_person_detections(
    frame: np.ndarray,
    detections: Sequence[Detection],
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    output = frame.copy()

    for detection in detections:
        x1, y1, x2, y2 = map(int, detection.box)
        label = f"{detection.label} {detection.confidence:.2f}"

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            output,
            label,
            (x1, max(y1 - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return output


def save_sample_detection(
    video_path: str,
    output_path: str,
    model_path: str = "yolov8n.pt",
    frame_index: int = 0,
) -> List[Detection]:
    detector = PersonDetector(model_path=model_path)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Cannot read frame {frame_index} from {video_path}")

        detections = detector.detect(frame)
        output_frame = draw_person_detections(frame, detections)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), output_frame)
        return detections
    finally:
        cap.release()