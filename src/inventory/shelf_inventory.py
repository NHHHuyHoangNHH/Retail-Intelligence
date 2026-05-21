from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import yaml

from detection.types import Detection


Point = Tuple[float, float]


@dataclass(frozen=True)
class ShelfZone:
    id: str
    name: str
    polygon: Tuple[Point, ...]
    min_product_count: int = 0
    max_empty_count: int = 0

    def contains(self, point: Point) -> bool:
        contour = np.array(self.polygon, dtype=np.float32)
        return cv2.pointPolygonTest(contour, point, False) >= 0


@dataclass(frozen=True)
class ShelfInventorySummary:
    shelf: ShelfZone
    product_count: int
    empty_count: int
    detections: Tuple[Detection, ...]

    @property
    def needs_restock(self) -> bool:
        if self.product_count < self.shelf.min_product_count:
            return True
        return self.empty_count > self.shelf.max_empty_count

    @property
    def alert_reason(self) -> str:
        reasons = []
        if self.product_count < self.shelf.min_product_count:
            reasons.append(
                f"product_count<{self.shelf.min_product_count}"
            )
        if self.empty_count > self.shelf.max_empty_count:
            reasons.append(f"empty_count>{self.shelf.max_empty_count}")
        return ", ".join(reasons)


class ShelfInventoryAnalyzer:
    def __init__(self, shelf_config_path: str) -> None:
        self.shelves = load_shelf_zones(shelf_config_path)

    def analyze(
        self,
        detections: Sequence[Detection],
    ) -> List[ShelfInventorySummary]:
        summaries = []

        for shelf in self.shelves:
            shelf_detections = tuple(
                detection
                for detection in detections
                if shelf.contains(detection.center)
            )
            empty_count = sum(
                1 for detection in shelf_detections if is_empty_label(detection.label)
            )
            product_count = len(shelf_detections) - empty_count

            summaries.append(
                ShelfInventorySummary(
                    shelf=shelf,
                    product_count=product_count,
                    empty_count=empty_count,
                    detections=shelf_detections,
                )
            )

        return summaries


def load_shelf_zones(shelf_config_path: str) -> List[ShelfZone]:
    path = Path(shelf_config_path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find shelf config: {shelf_config_path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    shelves = []
    for shelf_data in data.get("shelves", []):
        polygon = tuple(
            (float(point[0]), float(point[1]))
            for point in shelf_data.get("polygon", [])
        )
        if len(polygon) < 3:
            raise ValueError(f"Shelf {shelf_data.get('id')} needs at least 3 points")

        shelves.append(
            ShelfZone(
                id=str(shelf_data["id"]),
                name=str(shelf_data.get("name", shelf_data["id"])),
                polygon=polygon,
                min_product_count=int(shelf_data.get("min_product_count", 0)),
                max_empty_count=int(shelf_data.get("max_empty_count", 0)),
            )
        )

    return shelves


def is_empty_label(label: str) -> bool:
    return "empty" in label.lower()


def draw_shelf_inventory(
    frame: np.ndarray,
    summaries: Sequence[ShelfInventorySummary],
) -> np.ndarray:
    output = frame.copy()

    for summary in summaries:
        color = (0, 0, 255) if summary.needs_restock else (0, 180, 0)
        polygon = np.array(summary.shelf.polygon, dtype=np.int32)
        cv2.polylines(output, [polygon], isClosed=True, color=color, thickness=2)

        x, y = polygon.min(axis=0)
        status = "RESTOCK" if summary.needs_restock else "OK"
        label = (
            f"{summary.shelf.id}: {status} "
            f"P{summary.product_count}/E{summary.empty_count}"
        )
        cv2.putText(
            output,
            label,
            (int(x), max(int(y) - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return output
