from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from detection.types import Detection


Point = Tuple[float, float]
Polygon = List[Point]


@dataclass(frozen=True)
class ShelfRoi:
    id: str
    name: str
    polygon: Polygon
    min_product_count: int = 3
    max_empty_count: int = 0


@dataclass
class ShelfInventorySummary:
    shelf: ShelfRoi
    product_count: int = 0
    empty_count: int = 0
    products: List[Detection] = field(default_factory=list)
    empty_spaces: List[Detection] = field(default_factory=list)

    @property
    def needs_restock(self) -> bool:
        return (
            self.product_count < self.shelf.min_product_count
            or self.empty_count > self.shelf.max_empty_count
        )

    @property
    def alert_reason(self) -> str:
        reasons = []
        if self.product_count < self.shelf.min_product_count:
            reasons.append(
                f"product_count={self.product_count} < min={self.shelf.min_product_count}"
            )
        if self.empty_count > self.shelf.max_empty_count:
            reasons.append(
                f"empty_count={self.empty_count} > max={self.shelf.max_empty_count}"
            )
        return "; ".join(reasons)


class ShelfInventoryAnalyzer:
    def __init__(self, shelf_config_path: str) -> None:
        self.shelves = load_shelf_rois(shelf_config_path)

    def analyze(self, detections: Sequence[Detection]) -> List[ShelfInventorySummary]:
        summaries = [
            ShelfInventorySummary(shelf=shelf)
            for shelf in self.shelves
        ]
        by_id: Dict[str, ShelfInventorySummary] = {
            summary.shelf.id: summary
            for summary in summaries
        }

        for detection in detections:
            shelf = self.find_shelf_for_detection(detection)
            if shelf is None:
                continue

            summary = by_id[shelf.id]
            if is_empty_shelf_detection(detection):
                summary.empty_count += 1
                summary.empty_spaces.append(detection)
            else:
                summary.product_count += 1
                summary.products.append(detection)

        return summaries

    def find_shelf_for_detection(self, detection: Detection) -> Optional[ShelfRoi]:
        center = detection.center
        matching_shelves = [
            shelf
            for shelf in self.shelves
            if point_in_polygon(center, shelf.polygon)
        ]

        if matching_shelves:
            return matching_shelves[0]

        return self._nearest_shelf_by_box_overlap(detection)

    def _nearest_shelf_by_box_overlap(self, detection: Detection) -> Optional[ShelfRoi]:
        best_shelf = None
        best_overlap = 0.0
        det_box = detection.box

        for shelf in self.shelves:
            shelf_box = polygon_to_box(shelf.polygon)
            overlap = intersection_area(det_box, shelf_box)
            if overlap > best_overlap:
                best_overlap = overlap
                best_shelf = shelf

        return best_shelf if best_overlap > 0 else None


def load_shelf_rois(config_path: str) -> List[ShelfRoi]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Shelf config not found: {config_path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    shelves = []
    for item in data.get("shelves", []):
        shelves.append(
            ShelfRoi(
                id=str(item["id"]),
                name=str(item.get("name", item["id"])),
                polygon=[
                    (float(point[0]), float(point[1]))
                    for point in item["polygon"]
                ],
                min_product_count=int(item.get("min_product_count", 3)),
                max_empty_count=int(item.get("max_empty_count", 0)),
            )
        )

    if not shelves:
        raise ValueError(f"No shelves configured in {config_path}")

    return shelves


def is_empty_shelf_detection(detection: Detection) -> bool:
    label = detection.label.lower()
    return "empty" in label or "space" in label


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    contour = np.array(polygon, dtype=np.float32)
    return cv2.pointPolygonTest(contour, point, False) >= 0


def polygon_to_box(polygon: Polygon) -> Tuple[float, float, float, float]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def intersection_area(
    box_a: Tuple[float, float, float, float],
    box_b: Tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    return (x2 - x1) * (y2 - y1)


def draw_shelf_inventory(
    frame: np.ndarray,
    summaries: Sequence[ShelfInventorySummary],
) -> np.ndarray:
    output = frame.copy()

    for summary in summaries:
        color = (0, 0, 255) if summary.needs_restock else (0, 180, 0)
        polygon = np.array(summary.shelf.polygon, dtype=np.int32)

        cv2.polylines(output, [polygon], isClosed=True, color=color, thickness=2)

        x, y = polygon[0]
        status = "RESTOCK" if summary.needs_restock else "OK"
        text = (
            f"{summary.shelf.id}: {status} "
            f"products={summary.product_count} empty={summary.empty_count}"
        )

        cv2.putText(
            output,
            text,
            (int(x), max(int(y) - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    return output