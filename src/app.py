import argparse
import shutil
import uuid
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from main import (
    DEFAULT_INSIGHTFACE_ROOT,
    DEFAULT_PERSON_MODEL,
    DEFAULT_YOLO_WORLD_MODEL,
    PROJECT_ROOT,
    run_session_report_sample,
    run_shelf_calibration,
    write_shelf_calibration_preview,
    write_shelf_config,
)


APP_OUTPUT_DIR = PROJECT_ROOT / "src/data/outputs/gradio"
SHELF_CONFIDENCE = 0.01
SHELF_PROMPTS = "retail shelf,store shelf,product shelf"
FRAME_STRIDE = 3
INVENTORY_STRIDE = 10
CAMERA_CONFIG = ""


def process_video(
    video_path,
    process_percent,
):
    if not video_path:
        raise gr.Error("Upload a video first.")

    run_id = uuid.uuid4().hex[:10]
    run_dir = APP_OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    input_path = run_dir / f"input{Path(video_path).suffix or '.mp4'}"
    shutil.copy2(video_path, input_path)
    max_frames = max_frames_from_percent(
        input_path,
        process_percent,
        FRAME_STRIDE,
    )

    shelves_path = run_dir / "shelves_auto.yaml"
    preview_path = run_dir / "shelf_roi_preview.jpg"
    output_video_path = run_dir / "pipeline_output.mp4"
    report_path = run_dir / "session_report.json"
    events_path = run_dir / "shelf_interaction_events.csv"
    inventory_path = run_dir / "inventory_timeline.csv"

    calibration_args = build_args(
        video=str(input_path),
        output_config=str(shelves_path),
        preview_output=str(preview_path),
        shelf_conf=SHELF_CONFIDENCE,
        shelf_prompts=SHELF_PROMPTS,
        calibration_frames=3,
        calibration_stride=30,
    )
    calibration_note = "YOLO-World ROI calibration succeeded."
    try:
        shelves, raw_boxes, sampled_frames = run_shelf_calibration(calibration_args)
    except ValueError as exc:
        shelves = build_fallback_shelf(input_path)
        raw_boxes = 0
        sampled_frames = 0
        write_shelf_config(shelves_path, shelves)
        preview_frame = read_preview_frame(input_path)
        write_shelf_calibration_preview(preview_path, preview_frame, shelves)
        calibration_note = f"YOLO-World ROI calibration failed: {exc}. Used fallback ROI."

    report_args = build_args(
        video=str(input_path),
        shelves=str(shelves_path),
        output=str(output_video_path),
        report_output=str(report_path),
        events_output=str(events_path),
        inventory_output=str(inventory_path),
        max_frames=max_frames,
        frame_stride=FRAME_STRIDE,
        inventory_stride=INVENTORY_STRIDE,
    )
    reid_note = "Re-ID backend: OSNet."
    try:
        written, report = run_session_report_sample(report_args, output_video_path)
    except RuntimeError as exc:
        if "torchreid" not in str(exc).lower() and "osnet" not in str(exc).lower():
            raise
        report_args.reid_backend = "color"
        reid_note = f"OSNet Re-ID failed: {exc}. Fell back to color Re-ID."
        written, report = run_session_report_sample(report_args, output_video_path)

    summary = (
        f"Processed {written} frame(s).\n"
        f"Requested video coverage: {float(process_percent):.0f}%.\n"
        f"{reid_note}\n"
        f"{calibration_note}\n"
        f"Auto ROI: {len(shelves)} shelf ROI(s) from {raw_boxes} YOLO-World box(es) "
        f"across {sampled_frames} sampled frame(s).\n"
        f"Unique customers: {report['single_visit_analytics']['unique_customers']}."
    )

    return (
        str(output_video_path),
        str(output_video_path),
        str(preview_path),
        str(report_path),
        str(events_path),
        str(inventory_path),
        summary,
    )


def build_args(**overrides):
    defaults = {
        "video": "",
        "output": "",
        "model": DEFAULT_PERSON_MODEL,
        "model_product": None,
        "shelves": "",
        "tracker": "botsort.yaml",
        "camera_id": "camera_1",
        "camera_config": CAMERA_CONFIG,
        "reid_backend": "osnet",
        "demographics_backend": "heuristic",
        "insightface_root": DEFAULT_INSIGHTFACE_ROOT,
        "trail_length": 45,
        "frame_stride": 3,
        "max_frames": 100,
        "interaction_min_dwell": 1.0,
        "interaction_cooldown": 3.0,
        "events_output": "",
        "report_output": "",
        "inventory_output": "",
        "inventory_stride": 10,
        "inventory_assignment_window": 5.0,
        "output_config": "",
        "preview_output": "",
        "calibration_frames": 3,
        "calibration_stride": 30,
        "yolo_world_model": DEFAULT_YOLO_WORLD_MODEL,
        "shelf_prompts": SHELF_PROMPTS,
        "shelf_conf": SHELF_CONFIDENCE,
        "merge_iou": 0.15,
        "roi_padding": 35,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def max_frames_from_percent(video_path: Path, process_percent, frame_stride: int):
    percent = max(1.0, min(float(process_percent), 100.0))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise gr.Error(f"Cannot open uploaded video: {video_path}")
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()

    if total_frames <= 0 or percent >= 100.0:
        return None

    target_source_frames = max(1, int(total_frames * percent / 100.0))
    return max(1, int(np.ceil(target_source_frames / frame_stride)))


def build_fallback_shelf(video_path: Path):
    frame = read_preview_frame(video_path)
    height, width = frame.shape[:2]
    x1 = int(width * 0.15)
    x2 = int(width * 0.85)
    y1 = int(height * 0.12)
    y2 = int(height * 0.88)
    return [
        {
            "id": "shelf_1",
            "name": "Fallback shelf",
            "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            "min_product_count": 6,
            "max_empty_count": 1,
            "cluster_product_count": 0,
        }
    ]


def read_preview_frame(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise gr.Error(f"Cannot open uploaded video: {video_path}")
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise gr.Error(f"Cannot read uploaded video: {video_path}")
    return frame


def build_interface():
    with gr.Blocks(title="Retail Intelligence Pipeline") as demo:
        gr.Markdown("# Retail Intelligence Pipeline")
        gr.Markdown(
            "Upload a video. The app auto-calibrates shelf ROI with YOLO-World, "
            "then runs the full tracking, interaction, inventory, and report pipeline."
        )

        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Input video")
                process_percent = gr.Slider(
                    label="Video coverage (%)",
                    minimum=1,
                    maximum=100,
                    value=100,
                    step=1,
                )
                run_button = gr.Button("Run pipeline", variant="primary")

            with gr.Column():
                output_video = gr.Video(label="Processed video", format="mp4")
                output_video_file = gr.File(label="Processed video file")
                roi_preview = gr.Image(label="Auto ROI preview")
                summary = gr.Textbox(label="Run summary", lines=5)
                report_file = gr.File(label="Session report JSON")
                events_file = gr.File(label="Shelf interaction CSV")
                inventory_file = gr.File(label="Inventory timeline CSV")

        run_button.click(
            fn=process_video,
            inputs=[
                video_input,
                process_percent,
            ],
            outputs=[
                output_video,
                output_video_file,
                roi_preview,
                report_file,
                events_file,
                inventory_file,
                summary,
            ],
        )

    return demo


if __name__ == "__main__":
    build_interface().launch()
