from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from inventory.shelf_inventory import ShelfZone, load_shelf_zones
from tracking.person_tracker import PersonTrack


Point = Tuple[float, float]
TrackShelfKey = Tuple[int, str]


@dataclass(frozen=True)
class ShelfInteractionEvent:
    type: str
    global_id: int
    local_track_id: int
    camera_id: str
    shelf_id: str
    frame_id: int
    timestamp_seconds: float
    dwell_seconds: float
    interaction_point: Point


class ShelfInteractionAnalyzer:
    def __init__(
        self,
        shelf_config_path: str,
        min_dwell_seconds: float = 1.0,
        cooldown_seconds: float = 3.0,
    ) -> None:
        self.shelf_zones = load_shelf_zones(shelf_config_path)
        self.min_dwell_seconds = min_dwell_seconds
        self.cooldown_seconds = cooldown_seconds
        self._active_since: Dict[TrackShelfKey, float] = {}
        self._last_event_at: Dict[TrackShelfKey, float] = {}

    def analyze(self, tracks: Sequence[PersonTrack]) -> List[ShelfInteractionEvent]:
        events: List[ShelfInteractionEvent] = []
        visible_keys = set()

        for track in tracks:
            interaction_point = self._estimate_interaction_point(track.box)

            for shelf in self.shelf_zones:
                key = (track.global_id, shelf.id)
                if not shelf.contains(interaction_point):
                    continue

                visible_keys.add(key)
                active_since = self._active_since.setdefault(
                    key,
                    track.timestamp_seconds,
                )
                dwell_seconds = track.timestamp_seconds - active_since
                if dwell_seconds < self.min_dwell_seconds:
                    continue

                last_event_at = self._last_event_at.get(key)
                if (
                    last_event_at is not None
                    and track.timestamp_seconds - last_event_at < self.cooldown_seconds
                ):
                    continue

                self._last_event_at[key] = track.timestamp_seconds
                events.append(
                    ShelfInteractionEvent(
                        type="shelf_interaction",
                        global_id=track.global_id,
                        local_track_id=track.local_track_id,
                        camera_id=track.camera_id,
                        shelf_id=shelf.id,
                        frame_id=track.frame_id,
                        timestamp_seconds=track.timestamp_seconds,
                        dwell_seconds=dwell_seconds,
                        interaction_point=interaction_point,
                    )
                )

        self._expire_inactive(visible_keys)
        return events

    def _expire_inactive(self, visible_keys: set[TrackShelfKey]) -> None:
        for key in list(self._active_since):
            if key not in visible_keys:
                del self._active_since[key]

    @staticmethod
    def _estimate_interaction_point(
        person_box: Tuple[float, float, float, float],
    ) -> Point:
        x1, y1, x2, y2 = person_box
        return ((x1 + x2) / 2.0, y1 + (y2 - y1) * 0.55)


def draw_shelf_interactions(
    frame: np.ndarray,
    shelves: Sequence[ShelfZone],
    tracks: Sequence[PersonTrack],
    events: Optional[Sequence[ShelfInteractionEvent]] = None,
) -> np.ndarray:
    output = frame.copy()
    event_keys = {
        (event.global_id, event.shelf_id)
        for event in events or []
    }

    for shelf in shelves:
        polygon = np.array(shelf.polygon, dtype=np.int32)
        cv2.polylines(output, [polygon], isClosed=True, color=(255, 160, 0), thickness=2)
        x, y = polygon.min(axis=0)
        cv2.putText(
            output,
            shelf.id,
            (int(x), max(int(y) - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 160, 0),
            2,
            cv2.LINE_AA,
        )

    for track in tracks:
        point = ShelfInteractionAnalyzer._estimate_interaction_point(track.box)
        point_xy = (int(point[0]), int(point[1]))
        color = (0, 0, 255) if any(
            key[0] == track.global_id for key in event_keys
        ) else (0, 255, 255)
        cv2.circle(output, point_xy, 5, color, -1, cv2.LINE_AA)

    for event in events or []:
        point_xy = (int(event.interaction_point[0]), int(event.interaction_point[1]))
        label = f"GID {event.global_id} -> {event.shelf_id}"
        cv2.putText(
            output,
            label,
            (point_xy[0] + 8, max(point_xy[1] - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return output
