# Retail Intelligence Pipeline

A retail camera video analysis prototype built with Gradio. Users upload any video, the system automatically finds shelf ROIs, runs customer tracking, Re-ID, shelf interaction analysis, inventory monitoring, pickup/return action estimation, and exports an overlay video along with reports.

## Key Features

- Upload video via Gradio.
- Automatically detect shelf ROIs for each video using YOLO-World.
- Track customers using YOLO + BoT-SORT.
- Re-identify customers using OSNet `osnet_x1_0`; falls back to color Re-ID if OSNet has dependency/model errors.
- Display on person bboxes: `ID`, estimated age/gender, and current action.
- Track inventory over time with restock alerts.
- Record pickup/return frequency per shelf.
- Export processed video, ROI preview image, JSON report, CSV interaction events, and CSV inventory timeline.

## File Structure

```text
retail-intelligence/
├── README.md
├── REPORT.md
├── configs/                         # Optional for legacy CLI; Gradio does not depend on this
├── src/
│   ├── app.py                       # Gradio app, main entrypoint
│   ├── main.py                      # Pipeline core and CLI utilities
│   ├── requirements.txt             # Main dependencies
│   ├── action/
│   │   └── shelf_interaction.py     # Detects customer interactions with shelf ROIs
│   ├── analytics/
│   │   ├── demographics.py          # Demographics heuristic for overlay/report
│   │   └── session_report.py        # Analytics/report aggregation
│   ├── detection/
│   │   ├── person_detector.py       # Person detection sample
│   │   ├── product_detector.py      # Product/empty-slot YOLO detector
│   │   └── types.py
│   ├── inventory/
│   │   └── shelf_inventory.py       # Product count, empty slot, restock alert
│   ├── tracking/
│   │   ├── person_tracker.py        # YOLO tracking + overlay
│   │   └── reid.py                  # Color Re-ID and OSNet Re-ID
│   ├── models/
│   │   ├── person-detection/
│   │   │   └── yolov8n.pt           # YOLO person model
│   │   ├── shelf-roi/
│   │   │   └── yolov8s-world.pt     # YOLO-World used for auto ROI
│   │   └── product-detection/
│   │       └── best.pt              # Product/empty-slot detector checkpoint
│   └── data/
│       ├── raw_videos/              # Sample videos
│       └── outputs/
│           └── gradio/<run_id>/     # Output per Gradio run
```

## Installation

Create a virtualenv and install dependencies:

```bash
cd retail-intelligence
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r src/requirements.txt
```

## Running Gradio

Start the app:

```bash
cd src
../.venv/bin/python app.py
```

Open the Gradio URL:

```text
http://127.0.0.1:7860
```

Workflow:

1. Upload a video.
2. Select `Video coverage (%)`.
3. Click `Run pipeline`.
4. View the output video and download report files.

Parameters currently hardcoded in [src/app.py](src/app.py):

- YOLO-World shelf confidence: `0.01`
- Shelf prompts: `retail shelf,store shelf,product shelf`
- Frame stride: `3`
- Inventory stride: `10`
- Re-ID backend: `osnet`, fallback `color`
- Demographics backend: `heuristic`

## Output

Each Gradio run creates a directory:

```text
src/data/outputs/gradio/<run_id>/
```

Main files:

- `input.<ext>`: uploaded video copied into the run folder.
- `shelves_auto.yaml`: auto-generated shelf ROI for that video.
- `shelf_roi_preview.jpg`: ROI preview image.
- `pipeline_output.mp4`: overlay video after pipeline run.
- `session_report.json`: aggregated analytics report.
- `shelf_interaction_events.csv`: customer shelf interaction events.
- `inventory_timeline.csv`: inventory/restock timeline.

## Models and Checkpoints

Models currently in use:

- Person detection/tracking: `src/models/person-detection/yolov8n.pt`.
- Auto shelf ROI: `src/models/shelf-roi/yolov8s-world.pt`.
- Product/empty-slot detection: `src/models/product-detection/best.pt` if available; otherwise the code can download a YOLO product detector checkpoint from Hugging Face.
- Re-ID: Torchreid OSNet `osnet_x1_0`.
- Demographics: heuristic.

Auto-download status:

- Product/empty-slot detector: auto-downloads via Hugging Face if `src/models/product-detection/best.pt` is missing.
- Person detector `yolov8n.pt`: auto-downloads via Ultralytics if missing.
- YOLO-World `yolov8s-world.pt`: auto-downloads via Ultralytics if missing.
- OSNet Re-ID: torchreid can auto-download pretrained weights when initializing `osnet_x1_0`.

Therefore, `.pt` checkpoints can be excluded from Git commits to keep the repo lightweight. An internet connection is required on the first run to download models to the correct directories.

## How the System Works

### 1. Single-Visit Customer Analytics

The system uses YOLO + BoT-SORT to track each customer within a video. Each track is assigned a `global_id` via Re-ID. The report aggregates:

- Number of unique customers.
- Appearance time and dwell time.
- Estimated age/gender distribution.
- Number of products picked up/returned per customer.
- Average number of products taken per visit.

This information is found in `session_report.json` under the `single_visit_analytics` section.

### 2. Correlation Between Customer Behavior and Product Placement

YOLO-World automatically finds shelf ROIs for each video. The system then checks whether each customer's interaction point falls within a shelf ROI. Each event is tagged with `shelf_id`, `global_id`, timestamp, and dwell time.

The report aggregates per shelf:

- Total interaction count.
- Number of unique customers who interacted.
- Pickup/return count.
- Ratio of interactions that led to a pickup.
- Shelf ranking by interaction level and conversion rate.

This information is found in `session_report.json` under the `customer_behavior_product_placement` section.

### 3. Real-Time Inventory Monitoring

During video processing, the pipeline samples inventory at `INVENTORY_STRIDE = 10`. The product detector identifies products and empty slots, then `ShelfInventoryAnalyzer` assigns detections to shelf ROIs.

The system outputs:

- Product count per shelf.
- Empty-slot count.
- Restock alerts.
- Inventory timeline.
- Pickup/return frequency.

The timeline is stored in `inventory_timeline.csv`, and the aggregated state is stored in `session_report.json`.

## Interview Q&A

### 1. How do you track and re-identify a customer across multiple cameras without face recognition?

The system does not use face recognition for Re-ID. The approach combines:

- Local tracker continuity within each camera using BoT-SORT.
- Body appearance embedding using OSNet `osnet_x1_0`.
- Fallback color histogram embedding if OSNet is unavailable.
- A global ID to link local tracks belonging to the same customer.
- For CLI multi-camera mode, a camera transition config can be used; in the current Gradio single-video mode, the camera config is left empty.

### 2. How do you distinguish a customer who picks up an item to buy versus one who just looks and puts it back?

The pipeline does not rely solely on the person's bounding box. It fuses two signal sources:

- Shelf interaction: whether the customer is standing in or interacting with a shelf ROI.
- Inventory delta: whether the product/empty-slot count changes after the interaction.

If the product count decreases or the empty-slot count increases after an interaction, the system records a `pickup`. If the product count increases or the empty-slot count decreases, the system records a `return`. Each event is assigned to the customer who most recently interacted with the same shelf within a given time window.

### 3. How do you maintain accuracy across different stores, camera angles, and lighting conditions?

The Gradio workflow handles camera angle differences through auto ROI:

- Each video runs YOLO-World independently to find shelf locations.
- ROIs are saved separately per run and are not hardcoded.
- Product/inventory counts are only computed within those ROIs.
- Re-ID uses body embeddings instead of faces, so it is less sensitive to face visibility.

For production use, improvements would include fine-tuning the detector on real store data, calibrating thresholds per camera, testing across different store layouts, and adding camera calibration for multi-camera setups.

### 4. What metrics do you use to benchmark inventory tracking and action recognition?

Inventory tracking:

- Product detection precision/recall/mAP.
- Empty-slot precision/recall/mAP.
- Count MAE/RMSE between predicted and manually counted values.
- Restock alert precision/recall.
- Restock alert latency.

Action recognition:

- Pickup/return precision, recall, F1.
- Event timestamp error.
- Customer-event assignment accuracy.
- Interaction-to-pickup conversion accuracy per shelf.

Tracking/Re-ID:

- IDF1.
- MOTA.
- HOTA.
- ID switches.

## 5. How does the tracking model handle heavy occlusions and overlapping bounding boxes when the store is crowded?

The current implementation handles moderate crowding with YOLO person detection plus BoT-SORT tracking. BoT-SORT keeps local track continuity by associating detections across frames using motion and box overlap, so a customer can usually keep the same local ID through short partial occlusions or brief bbox overlaps.

The pipeline also adds Re-ID on top of the local tracker. Each local track is mapped to a `global_id` using an OSNet body embedding, with a color-histogram fallback if OSNet is unavailable. This helps recover identity when a customer is temporarily lost and later appears again, especially when the body appearance is still visible.

There are still practical limits. If a person is fully hidden for a long period, or if multiple customers with similar clothing overlap heavily, the detector/tracker can drop the track or switch IDs. In that case, downstream shelf-interaction and pickup/return assignment can also become less reliable. For production, this should be improved and validated with crowded-store data by tuning tracker thresholds, using a stronger person detector/Re-ID model, adding camera-specific calibration, and monitoring tracking metrics such as IDF1, HOTA, MOTA, and ID switches.