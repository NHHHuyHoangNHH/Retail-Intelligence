from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from detection.types import Detection

try:
    from huggingface_hub import hf_hub_download
except ImportError as exc:
    hf_hub_download = None
    _HF_IMPORT_ERROR = exc
else:
    _HF_IMPORT_ERROR = None

try:
    from ultralytics import YOLO
except ImportError as exc:
    YOLO = None
    _YOLO_IMPORT_ERROR = exc
else:
    _YOLO_IMPORT_ERROR = None

DEFAULT_PRODUCT_MODEL_REPO = "foduucom/product-detection-in-shelf-yolov8"
DEFAULT_PRODUCT_MODEL_FILE = "best.pt"
DEFAULT_PRODUCT_CHECKPOINT_DIR = str(
    Path(__file__).resolve().parents[2] / "src/models/product-detection"
)
LOCAL_PRODUCT_MODEL_CANDIDATES = (
    Path(__file__).resolve().parents[2] / "src/models/product-detection/best.pt",
)


class ProductShelfDetector:
    def __init__(
        self,
        model_path: Optional[str] = None,
        repo_id: str = DEFAULT_PRODUCT_MODEL_REPO,
        filename: str = DEFAULT_PRODUCT_MODEL_FILE,
        checkpoint_dir: str = DEFAULT_PRODUCT_CHECKPOINT_DIR,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
        image_size: int = 640,
        max_det: int = 1000,
    ) -> None:
        if YOLO is None:
            raise RuntimeError(
                "Missing dependency: ultralytics. Install it with "
                "`pip install -r requirements.txt`."
            ) from _YOLO_IMPORT_ERROR

        self.model_path = model_path or self._ensure_checkpoint(
            repo_id=repo_id,
            filename=filename,
            checkpoint_dir=checkpoint_dir,
        )

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.image_size = image_size
        self.max_det = max_det
        self.model = YOLO(self.model_path)

    def _ensure_checkpoint(
        self,
        repo_id: str,
        filename: str,
        checkpoint_dir: str,
    ) -> str:
        for candidate in LOCAL_PRODUCT_MODEL_CANDIDATES:
            if candidate.exists():
                return str(candidate)

        local_path = Path(checkpoint_dir) / filename
        if local_path.exists():
            return str(local_path)

        if hf_hub_download is None:
            raise RuntimeError(
                "Missing dependency: huggingface_hub. Install it with "
                "`pip install -r requirements.txt`, or pass --model-product "
                "with a local best.pt path."
            ) from _HF_IMPORT_ERROR

        local_path.parent.mkdir(parents=True, exist_ok=True)
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_path.parent),
            local_dir_use_symlinks=False,
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if frame is None or frame.size == 0:
            return []

        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            max_det=self.max_det,
            verbose=False,
        )

        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        names = result.names or {}
        detections: List[Detection] = []

        if boxes is None:
            return detections

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = str(names.get(class_id, f"class_{class_id}"))

            detections.append(
                Detection(
                    box=(x1, y1, x2, y2),
                    confidence=confidence,
                    class_id=class_id,
                    label=label,
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


def draw_product_detections(
    frame: np.ndarray,
    detections: Sequence[Detection],
) -> np.ndarray:
    output = frame.copy()

    for detection in detections:
        x1, y1, x2, y2 = map(int, detection.box)
        color = _color_for_label(detection.label)
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


def _color_for_label(label: str) -> Tuple[int, int, int]:
    lowered = label.lower()
    if "empty" in lowered:
        return (0, 0, 255)
    return (255, 180, 0)


def save_sample_product_detection(
    video_path: str,
    output_path: str,
    model_path: Optional[str] = None,
    frame_index: int = 0,
) -> List[Detection]:
    detector = ProductShelfDetector(model_path=model_path)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Cannot read frame {frame_index} from {video_path}")

        detections = detector.detect(frame)
        output_frame = draw_product_detections(frame, detections)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), output_frame)
        return detections
    finally:
        cap.release()
