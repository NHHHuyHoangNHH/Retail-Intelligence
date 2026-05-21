import argparse
import csv
import shutil
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLOWorld
from ultralytics.utils.downloads import attempt_download_asset

from action.shelf_interaction import (
    ShelfInteractionAnalyzer,
    draw_shelf_interactions,
)
from analytics.demographics import build_demographic_estimator
from analytics.session_report import (
    InventoryChangeTracker,
    SessionReportBuilder,
    write_inventory_timeline_csv,
    write_session_report,
)
from detection.person_detector import save_sample_detection
from detection.product_detector import (
    ProductShelfDetector,
    draw_product_detections,
)
from inventory.shelf_inventory import ShelfInventoryAnalyzer, draw_shelf_inventory
from tracking.person_tracker import (
    PersonTracker,
    draw_person_tracks,
    save_tracking_video,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = str(PROJECT_ROOT / "src/data/raw_videos/1.mp4")
PERSON_MODEL_FILENAME = "yolov8n.pt"
YOLO_WORLD_MODEL_FILENAME = "yolov8s-world.pt"
DEFAULT_PERSON_MODEL = str(
    PROJECT_ROOT / f"src/models/person-detection/{PERSON_MODEL_FILENAME}"
)
DEFAULT_PERSON_OUTPUT = str(PROJECT_ROOT / "src/data/outputs/person_detection_sample.jpg")
DEFAULT_PRODUCT_OUTPUT = str(PROJECT_ROOT / "src/data/outputs/product_detection_sample.jpg")
DEFAULT_TRACKING_OUTPUT = str(PROJECT_ROOT / "src/data/outputs/person_tracking_sample.mp4")
DEFAULT_INTERACTION_OUTPUT = str(PROJECT_ROOT / "src/data/outputs/shelf_interaction_sample.mp4")
DEFAULT_INTERACTION_EVENTS = str(PROJECT_ROOT / "src/data/outputs/shelf_interaction_events.csv")
DEFAULT_REPORT_OUTPUT = str(PROJECT_ROOT / "src/data/outputs/session_report.json")
DEFAULT_REPORT_VIDEO_OUTPUT = str(PROJECT_ROOT / "src/data/outputs/session_report_overlay.mp4")
DEFAULT_INVENTORY_OUTPUT = str(PROJECT_ROOT / "src/data/outputs/inventory_timeline.csv")
DEFAULT_CALIBRATED_SHELVES = str(PROJECT_ROOT / "configs/shelves_auto.yaml")
DEFAULT_CALIBRATION_PREVIEW = str(PROJECT_ROOT / "src/data/outputs/shelf_calibration_preview.jpg")
DEFAULT_YOLO_WORLD_MODEL = str(
    PROJECT_ROOT / f"src/models/shelf-roi/{YOLO_WORLD_MODEL_FILENAME}"
)
DEFAULT_SHELVES = str(PROJECT_ROOT / "configs/shelves.yaml")
DEFAULT_CAMERAS = str(PROJECT_ROOT / "configs/cameras.yaml")
DEFAULT_INSIGHTFACE_ROOT = str(PROJECT_ROOT / "src/models/insightface")


def ensure_ultralytics_checkpoint(local_path: str, asset_name: str) -> str:
    path = Path(local_path)
    if path.exists():
        return str(path)

    if path.name != asset_name:
        raise FileNotFoundError(f"Cannot find model checkpoint: {local_path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    downloaded_path = Path(attempt_download_asset(asset_name))
    if not downloaded_path.exists():
        raise FileNotFoundError(f"Cannot download model checkpoint: {asset_name}")

    if downloaded_path.resolve() != path.resolve():
        shutil.copy2(downloaded_path, path)

    return str(path)


def ensure_person_model(model_path: str) -> str:
    return ensure_ultralytics_checkpoint(model_path, PERSON_MODEL_FILENAME)


def ensure_yolo_world_model(model_path: str) -> str:
    return ensure_ultralytics_checkpoint(model_path, YOLO_WORLD_MODEL_FILENAME)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retail intelligence demo runner")
    parser.add_argument(
        "--task",
        choices=(
            "person",
            "product",
            "tracking",
            "interaction",
            "report",
            "calibrate-shelves",
        ),
        default="person",
        help="Run detection, tracking, report, or automatic shelf ROI calibration.",
    )
    parser.add_argument(
        "--video",
        default=DEFAULT_VIDEO,
        help="Path to the input video.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_PERSON_OUTPUT,
        help="Path to save the visualized frame.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_PERSON_MODEL,
        help="Person YOLO model path.",
    )
    parser.add_argument(
        "--model-product",
        default=None,
        help="Optional local product/shelf checkpoint path. If omitted, best.pt is downloaded from Hugging Face.",
    )
    parser.add_argument(
        "--shelves",
        default=DEFAULT_SHELVES,
        help="Shelf ROI config used by the product inventory task.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index used for the quick detection sample.",
    )
    parser.add_argument(
        "--tracker",
        default="botsort.yaml",
        choices=("botsort.yaml", "bytetrack.yaml"),
        help="Ultralytics tracker config for person tracking.",
    )
    parser.add_argument(
        "--camera-id",
        default="camera_1",
        help="Camera identifier used for cross-camera re-identification.",
    )
    parser.add_argument(
        "--camera-config",
        default=DEFAULT_CAMERAS,
        help="Camera transition config used by cross-camera re-identification.",
    )
    parser.add_argument(
        "--reid-backend",
        choices=("color", "osnet"),
        default="color",
        help="Appearance embedding backend. OSNet requires optional torchreid install.",
    )
    parser.add_argument(
        "--demographics-backend",
        choices=("heuristic", "buffalo_l", "none"),
        default="heuristic",
        help="Demographics backend used in report mode.",
    )
    parser.add_argument(
        "--insightface-root",
        default=DEFAULT_INSIGHTFACE_ROOT,
        help="InsightFace model root. buffalo_l is expected under this directory.",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=45,
        help="Number of recent positions to draw as a movement trail.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Process every Nth frame for tracking videos.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of processed frames for tracking.",
    )
    parser.add_argument(
        "--interaction-min-dwell",
        type=float,
        default=1.0,
        help="Minimum seconds near a shelf before emitting an interaction event.",
    )
    parser.add_argument(
        "--interaction-cooldown",
        type=float,
        default=3.0,
        help="Minimum seconds between repeated interaction events for one person/shelf.",
    )
    parser.add_argument(
        "--events-output",
        default=None,
        help="Optional CSV path for shelf interaction events.",
    )
    parser.add_argument(
        "--report-output",
        default=DEFAULT_REPORT_OUTPUT,
        help="JSON path for the full session analytics report.",
    )
    parser.add_argument(
        "--inventory-output",
        default=None,
        help="Optional CSV path for real-time inventory snapshots.",
    )
    parser.add_argument(
        "--inventory-stride",
        type=int,
        default=15,
        help="Run product inventory every N processed frames in report mode.",
    )
    parser.add_argument(
        "--inventory-assignment-window",
        type=float,
        default=5.0,
        help="Seconds to associate inventory deltas with recent shelf interactions.",
    )
    parser.add_argument(
        "--output-config",
        default=DEFAULT_CALIBRATED_SHELVES,
        help="YAML path written by calibrate-shelves.",
    )
    parser.add_argument(
        "--preview-output",
        default=DEFAULT_CALIBRATION_PREVIEW,
        help="Preview image written by calibrate-shelves.",
    )
    parser.add_argument(
        "--calibration-frames",
        type=int,
        default=8,
        help="Number of frames sampled for YOLO-World shelf calibration.",
    )
    parser.add_argument(
        "--calibration-stride",
        type=int,
        default=30,
        help="Frame stride used for YOLO-World shelf calibration.",
    )
    parser.add_argument(
        "--yolo-world-model",
        default=DEFAULT_YOLO_WORLD_MODEL,
        help="YOLO-World checkpoint used by calibrate-shelves.",
    )
    parser.add_argument(
        "--shelf-prompts",
        default="retail shelf,store shelf,shelf,display rack,product shelf",
        help="Comma-separated YOLO-World class prompts for shelf ROI detection.",
    )
    parser.add_argument(
        "--shelf-conf",
        type=float,
        default=0.08,
        help="YOLO-World confidence threshold for shelf ROI detection.",
    )
    parser.add_argument(
        "--merge-iou",
        type=float,
        default=0.25,
        help="IoU threshold used to merge repeated shelf ROI detections.",
    )
    parser.add_argument(
        "--roi-padding",
        type=int,
        default=35,
        help="Pixels added around each auto-calibrated shelf ROI.",
    )
    return parser.parse_args()


def run_product_inventory_sample(args: argparse.Namespace, output_path: Path):
    detector = ProductShelfDetector(model_path=args.model_product)
    inventory_analyzer = ShelfInventoryAnalyzer(args.shelves)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Cannot read frame {args.frame} from {args.video}")

        detections = detector.detect(frame)
        summaries = inventory_analyzer.analyze(detections)

        output_frame = draw_product_detections(frame, detections)
        output_frame = draw_shelf_inventory(output_frame, summaries)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), output_frame)

        return detections, summaries
    finally:
        cap.release()


def run_shelf_calibration(args: argparse.Namespace):
    model_path = ensure_yolo_world_model(args.yolo_world_model)

    model = YOLOWorld(model_path)
    prompts = [
        prompt.strip()
        for prompt in args.shelf_prompts.split(",")
        if prompt.strip()
    ]
    model.set_classes(prompts)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    frame_id = 0
    sampled = 0
    boxes = []
    preview_frame = None

    try:
        while sampled < args.calibration_frames:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_id % max(args.calibration_stride, 1) == 0:
                if preview_frame is None:
                    preview_frame = frame.copy()
                boxes.extend(
                    detect_shelf_boxes_with_yolo_world(
                        model=model,
                        frame=frame,
                        confidence_threshold=args.shelf_conf,
                    )
                )
                sampled += 1

            frame_id += 1
    finally:
        cap.release()

    if preview_frame is None:
        raise ValueError(f"Cannot sample calibration frames from {args.video}")

    shelves = build_shelf_rois_from_yolo_world(
        boxes=boxes,
        frame_shape=preview_frame.shape,
        merge_iou=args.merge_iou,
        padding=args.roi_padding,
    )
    if not shelves:
        raise ValueError(
            "No shelf ROI found. Try lowering --shelf-conf, adding prompts with "
            "--shelf-prompts, or using another YOLO-World checkpoint."
        )

    write_shelf_config(Path(args.output_config), shelves)
    write_shelf_calibration_preview(Path(args.preview_output), preview_frame, shelves)
    return shelves, len(boxes), sampled


def detect_shelf_boxes_with_yolo_world(
    model,
    frame,
    confidence_threshold: float,
):
    results = model.predict(
        frame,
        conf=confidence_threshold,
        verbose=False,
    )
    if not results:
        return []

    result = results[0]
    if result.boxes is None:
        return []

    boxes = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        boxes.append((float(x1), float(y1), float(x2), float(y2)))
    return boxes


def build_shelf_rois_from_yolo_world(
    boxes,
    frame_shape,
    merge_iou: float,
    padding: int,
):
    if not boxes:
        return []

    merged_boxes = merge_overlapping_boxes(boxes, merge_iou)
    merged_boxes = suppress_nested_boxes(merged_boxes, overlap_threshold=0.55)
    height, width = frame_shape[:2]
    min_area = width * height * 0.015
    shelves = []

    for box in sorted(merged_boxes, key=lambda value: value[0]):
        x1 = max(0, int(box[0] - padding))
        y1 = max(0, int(box[1] - padding))
        x2 = min(width - 1, int(box[2] + padding))
        y2 = min(height - 1, int(box[3] + padding))
        if x2 <= x1 or y2 <= y1:
            continue
        if (x2 - x1) * (y2 - y1) < min_area:
            continue

        shelves.append(
            {
                "id": f"shelf_{len(shelves) + 1}",
                "name": f"Auto shelf {len(shelves) + 1}",
                "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                "min_product_count": 6,
                "max_empty_count": 1,
                "cluster_product_count": 1,
            }
        )

    return shelves


def merge_overlapping_boxes(boxes, iou_threshold: float):
    merged = []
    for box in boxes:
        current = box
        matched_idx = None
        for idx, existing in enumerate(merged):
            if boxes_should_merge(current, existing, iou_threshold):
                matched_idx = idx
                current = union_box(current, existing)
                break
        if matched_idx is None:
            merged.append(current)
        else:
            merged[matched_idx] = current

    changed = True
    while changed:
        changed = False
        compacted = []
        for box in merged:
            match = None
            for idx, existing in enumerate(compacted):
                if boxes_should_merge(box, existing, iou_threshold):
                    match = idx
                    break
            if match is None:
                compacted.append(box)
            else:
                compacted[match] = union_box(compacted[match], box)
                changed = True
        merged = compacted

    return merged


def boxes_should_merge(a, b, iou_threshold: float) -> bool:
    return (
        box_iou(a, b) >= iou_threshold
        or box_intersection_over_smaller(a, b) >= 0.50
    )


def suppress_nested_boxes(boxes, overlap_threshold: float):
    boxes_by_area = sorted(boxes, key=box_area, reverse=True)
    kept = []
    for box in boxes_by_area:
        if any(
            box_intersection_over_smaller(box, kept_box) >= overlap_threshold
            for kept_box in kept
        ):
            continue
        kept.append(box)
    return kept


def box_area(box) -> float:
    return max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)


def box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def box_intersection_over_smaller(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    smaller = min(area_a, area_b)
    if smaller <= 0:
        return 0.0
    return inter / smaller


def union_box(a, b):
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )


def write_shelf_config(output_path: Path, shelves) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "shelves": [
            {
                "id": shelf["id"],
                "name": shelf["name"],
                "polygon": shelf["polygon"],
                "min_product_count": shelf["min_product_count"],
                "max_empty_count": shelf["max_empty_count"],
            }
            for shelf in shelves
        ]
    }
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def write_shelf_calibration_preview(output_path: Path, frame, shelves) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview = frame.copy()
    for shelf in shelves:
        polygon = np.array(shelf["polygon"], dtype=np.int32)
        cv2.polylines(preview, [polygon], isClosed=True, color=(0, 255, 255), thickness=3)
        x, y = polygon.min(axis=0)
        label = f"{shelf['id']} products={shelf['cluster_product_count']}"
        cv2.putText(
            preview,
            label,
            (int(x), max(int(y) - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output_path), preview)


def run_shelf_interaction_sample(args: argparse.Namespace, output_path: Path):
    tracker = PersonTracker(
        model_path=args.model,
        tracker_config=args.tracker,
        camera_id=args.camera_id,
        reid_backend=args.reid_backend,
        camera_config_path=args.camera_config,
    )
    interaction_analyzer = ShelfInteractionAnalyzer(
        shelf_config_path=args.shelves,
        min_dwell_seconds=args.interaction_min_dwell,
        cooldown_seconds=args.interaction_cooldown,
    )

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / max(args.frame_stride, 1),
        (width, height),
    )

    all_events = []
    written = 0

    try:
        for _, frame, tracks in tracker.track_video(
            video_path=args.video,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
        ):
            events = interaction_analyzer.analyze(tracks)
            all_events.extend(events)

            output_frame = draw_person_tracks(frame, tracks)
            output_frame = draw_shelf_interactions(
                output_frame,
                interaction_analyzer.shelf_zones,
                tracks,
                events,
            )
            writer.write(output_frame)
            written += 1
    finally:
        writer.release()

    if args.events_output:
        write_interaction_events_csv(Path(args.events_output), all_events)

    return written, all_events


def write_interaction_events_csv(output_path: Path, events) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "type",
                "global_id",
                "local_track_id",
                "camera_id",
                "shelf_id",
                "frame_id",
                "timestamp_seconds",
                "dwell_seconds",
                "interaction_x",
                "interaction_y",
            ],
        )
        writer.writeheader()
        for event in events:
            writer.writerow(
                {
                    "type": event.type,
                    "global_id": event.global_id,
                    "local_track_id": event.local_track_id,
                    "camera_id": event.camera_id,
                    "shelf_id": event.shelf_id,
                    "frame_id": event.frame_id,
                    "timestamp_seconds": f"{event.timestamp_seconds:.3f}",
                    "dwell_seconds": f"{event.dwell_seconds:.3f}",
                    "interaction_x": f"{event.interaction_point[0]:.1f}",
                    "interaction_y": f"{event.interaction_point[1]:.1f}",
                }
            )


def format_demographic_overlay(observation) -> str:
    age = observation.apparent_age_group.replace("_", "-")
    gender = observation.gender.replace("_", "-")
    return f"Gender:{gender} Age:{age}"


def action_for_track(
    track,
    interactions,
    action_state,
    timestamp_seconds: float,
    hold_seconds: float = 3.0,
) -> str:
    state = action_state.get(track.global_id)
    if state and timestamp_seconds - state["timestamp_seconds"] <= hold_seconds:
        return state["label"]

    if any(event.global_id == track.global_id for event in interactions):
        return "Browsing"

    return "Walking"


def draw_dashboard(
    frame,
    fps: float,
    total_customers: int,
    demographic_labels,
    picked_quantity: int,
    returned_quantity: int,
    inventory_summaries,
) -> None:
    overlay = frame.copy()
    panel_width = min(250, frame.shape[1])
    panel_height = min(132, frame.shape[0])
    panel_x = 0
    panel_y = frame.shape[0] - panel_height
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_width, panel_y + panel_height),
        (20, 20, 20),
        -1,
    )
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    male_count = sum(1 for label in demographic_labels.values() if "/male" in label)
    female_count = sum(1 for label in demographic_labels.values() if "/female" in label)
    unknown_count = max(total_customers - male_count - female_count, 0)

    lines = [
        f"FPS: {fps:.1f}",
        f"Customers: {total_customers}",
        f"M:{male_count} F:{female_count} U:{unknown_count}",
        f"Pick:{picked_quantity} Return:{returned_quantity}",
    ]

    for summary in inventory_summaries[:2]:
        status = "RESTOCK" if summary.needs_restock else "OK"
        lines.append(
            f"{summary.shelf.id}: {summary.product_count} | {status}"
        )

    for idx, line in enumerate(lines[:6]):
        color = (0, 100, 255) if "RESTOCK" in line else (200, 255, 200)
        cv2.putText(
            frame,
            line,
            (panel_x + 8, panel_y + 18 + idx * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def run_session_report_sample(args: argparse.Namespace, output_path: Path):
    args.model = ensure_person_model(args.model)
    tracker = PersonTracker(
        model_path=args.model,
        tracker_config=args.tracker,
        camera_id=args.camera_id,
        reid_backend=args.reid_backend,
        camera_config_path=args.camera_config,
    )
    product_detector = ProductShelfDetector(model_path=args.model_product)
    inventory_analyzer = ShelfInventoryAnalyzer(args.shelves)
    interaction_analyzer = ShelfInteractionAnalyzer(
        shelf_config_path=args.shelves,
        min_dwell_seconds=args.interaction_min_dwell,
        cooldown_seconds=args.interaction_cooldown,
    )
    inventory_change_tracker = InventoryChangeTracker(
        assignment_window_seconds=args.inventory_assignment_window,
    )
    demographic_estimator = build_demographic_estimator(
        backend=args.demographics_backend,
        model_root=args.insightface_root,
    )
    report_builder = SessionReportBuilder(
        demographic_estimator=demographic_estimator,
    )

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = create_video_writer(
        output_path,
        fps / max(args.frame_stride, 1),
        (width, height),
    )

    written = 0
    frame_interval_seconds = max(args.frame_stride, 1) / fps
    last_inventory_summaries = []
    all_interactions = []
    picked_quantity_by_customer = defaultdict(int)
    returned_quantity_by_customer = defaultdict(int)
    latest_demographic_labels = {}
    action_state_by_customer = {}
    started_at = time.time()

    try:
        for processed_index, (frame_id, frame, tracks) in enumerate(
            tracker.track_video(
                video_path=args.video,
                frame_stride=args.frame_stride,
                max_frames=args.max_frames,
            )
        ):
            timestamp_seconds = frame_id / fps
            demographic_observations = report_builder.add_tracks(
                tracks,
                frame_interval_seconds,
                frame,
            )
            demographic_labels = {
                observation.global_id: format_demographic_overlay(observation)
                for observation in demographic_observations
            }
            latest_demographic_labels.update(demographic_labels)
            report_builder.add_shelf_presence(
                tracks,
                interaction_analyzer.shelf_zones,
                frame_interval_seconds,
            )

            interactions = interaction_analyzer.analyze(tracks)
            all_interactions.extend(interactions)
            report_builder.add_interactions(interactions)
            inventory_change_tracker.update_interactions(
                interactions,
                timestamp_seconds,
            )

            detections = []
            if processed_index % max(args.inventory_stride, 1) == 0:
                detections = product_detector.detect(frame)
                last_inventory_summaries = inventory_analyzer.analyze(detections)
                report_builder.add_inventory_snapshots(
                    summaries=last_inventory_summaries,
                    frame_id=frame_id,
                    timestamp_seconds=timestamp_seconds,
                )
                inventory_events = inventory_change_tracker.update_inventory(
                    summaries=last_inventory_summaries,
                    frame_id=frame_id,
                    timestamp_seconds=timestamp_seconds,
                )
                for event in inventory_events:
                    if event.assigned_global_id is None:
                        continue
                    if event.type == "pickup":
                        picked_quantity_by_customer[event.assigned_global_id] += (
                            event.quantity
                        )
                        action_state_by_customer[event.assigned_global_id] = {
                            "label": "Pick Up / Holding",
                            "timestamp_seconds": timestamp_seconds,
                        }
                    elif event.type == "return":
                        returned_quantity_by_customer[event.assigned_global_id] += (
                            event.quantity
                        )
                        action_state_by_customer[event.assigned_global_id] = {
                            "label": "Return / Not Holding",
                            "timestamp_seconds": timestamp_seconds,
                        }
                report_builder.add_inventory_events(inventory_events)

            track_extra_labels = {}
            for track in tracks:
                parts = []
                demographic_label = latest_demographic_labels.get(track.global_id)
                if demographic_label:
                    parts.append(demographic_label)
                action = action_for_track(
                    track,
                    interactions,
                    action_state_by_customer,
                    timestamp_seconds,
                )
                parts.append(f"Action:{action}")
                track_extra_labels[track.global_id] = " | ".join(parts)

            output_frame = draw_person_tracks(
                frame,
                tracks,
                extra_labels=track_extra_labels,
            )
            output_frame = draw_shelf_interactions(
                output_frame,
                interaction_analyzer.shelf_zones,
                tracks,
                interactions,
            )
            if detections:
                output_frame = draw_product_detections(output_frame, detections)
            if last_inventory_summaries:
                output_frame = draw_shelf_inventory(
                    output_frame,
                    last_inventory_summaries,
                )
            elapsed = time.time() - started_at
            current_fps = (written + 1) / max(elapsed, 1e-6)
            draw_dashboard(
                output_frame,
                fps=current_fps,
                total_customers=len(latest_demographic_labels),
                demographic_labels=latest_demographic_labels,
                picked_quantity=sum(picked_quantity_by_customer.values()),
                returned_quantity=sum(returned_quantity_by_customer.values()),
                inventory_summaries=last_inventory_summaries,
            )
            writer.write(output_frame)
            written += 1
    finally:
        writer.release()

    report = report_builder.build()
    write_session_report(Path(args.report_output), report)

    if args.inventory_output:
        write_inventory_timeline_csv(
            Path(args.inventory_output),
            report["real_time_inventory_monitoring"]["timeline"],
        )

    if args.events_output:
        write_interaction_events_csv(Path(args.events_output), all_interactions)

    return written, report


def create_video_writer(output_path: Path, fps: float, size):
    for codec in ("avc1", "H264", "mp4v"):
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            size,
        )
        if writer.isOpened():
            return writer
        writer.release()

    raise RuntimeError(f"Cannot create video writer for {output_path}")


def main() -> None:
    args = parse_args()
    args.model = ensure_person_model(args.model)

    output_path = Path(args.output)

    if args.task == "person":
        detections = save_sample_detection(
            video_path=args.video,
            output_path=str(output_path),
            model_path=args.model,
            frame_index=args.frame,
        )
    elif args.task == "product":
        if args.output == DEFAULT_PERSON_OUTPUT:
            output_path = Path(DEFAULT_PRODUCT_OUTPUT)

        detections, summaries = run_product_inventory_sample(args, output_path)

        for summary in summaries:
            status = "RESTOCK" if summary.needs_restock else "OK"
            reason = f" ({summary.alert_reason})" if summary.needs_restock else ""
            print(
                f"{summary.shelf.id}: {status} "
                f"products={summary.product_count} "
                f"empty={summary.empty_count}{reason}"
            )
    elif args.task == "calibrate-shelves":
        shelves, product_boxes, sampled_frames = run_shelf_calibration(args)
        print(
            f"Calibrated {len(shelves)} shelf ROI(s) from "
            f"{product_boxes} product box(es) across {sampled_frames} frame(s)"
        )
        print(f"Wrote shelf config to {args.output_config}")
        print(f"Wrote preview image to {args.preview_output}")
        return
    elif args.task == "tracking":
        if args.output == DEFAULT_PERSON_OUTPUT:
            output_path = Path(DEFAULT_TRACKING_OUTPUT)

        written = save_tracking_video(
            video_path=args.video,
            output_path=str(output_path),
            model_path=args.model,
            tracker_config=args.tracker,
            camera_id=args.camera_id,
            reid_backend=args.reid_backend,
            camera_config_path=args.camera_config,
            trail_length=args.trail_length,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
        )
        detections = []
        print(f"Wrote {written} tracked frame(s)")
    elif args.task == "interaction":
        if args.output == DEFAULT_PERSON_OUTPUT:
            output_path = Path(DEFAULT_INTERACTION_OUTPUT)
        if args.events_output is None:
            args.events_output = DEFAULT_INTERACTION_EVENTS

        written, events = run_shelf_interaction_sample(args, output_path)
        detections = events
        print(f"Wrote {written} interaction frame(s)")
        print(f"Wrote {len(events)} interaction event(s) to {args.events_output}")
    else:
        if args.output == DEFAULT_PERSON_OUTPUT:
            output_path = Path(DEFAULT_REPORT_VIDEO_OUTPUT)
        if args.events_output is None:
            args.events_output = DEFAULT_INTERACTION_EVENTS
        if args.inventory_output is None:
            args.inventory_output = DEFAULT_INVENTORY_OUTPUT

        written, report = run_session_report_sample(args, output_path)
        detections = report["inventory_action_recognition"]["events"]
        print(f"Wrote {written} report frame(s)")
        print(f"Wrote session report to {args.report_output}")
        print(f"Wrote interaction events to {args.events_output}")
        print(f"Wrote inventory timeline to {args.inventory_output}")

    print(f"Detected {len(detections)} object(s)")
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
