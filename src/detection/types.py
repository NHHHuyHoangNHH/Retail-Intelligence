from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Detection:
    """One object detected in a frame."""

    box: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    label: str

    @property
    def xyxy(self) -> Tuple[float, float, float, float]:
        return self.box

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)