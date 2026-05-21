from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import json
import numpy as np

from action.shelf_interaction import ShelfInteractionEvent
from analytics.demographics import (
    DemographicObservation,
    HeuristicDemographicEstimator,
)
from inventory.shelf_inventory import ShelfInventorySummary, ShelfZone
from tracking.person_tracker import PersonTrack


@dataclass(frozen=True)
class InventoryChangeEvent:
    type: str
    shelf_id: str
    frame_id: int
    timestamp_seconds: float
    quantity: int
    product_count_before: int
    product_count_after: int
    empty_count_before: int
    empty_count_after: int
    assigned_global_id: Optional[int] = None


@dataclass(frozen=True)
class InventorySnapshot:
    shelf_id: str
    frame_id: int
    timestamp_seconds: float
    product_count: int
    empty_count: int
    needs_restock: bool
    alert_reason: str


class InventoryChangeTracker:
    def __init__(self, assignment_window_seconds: float = 5.0) -> None:
        self.assignment_window_seconds = assignment_window_seconds
        self._previous_by_shelf: Dict[str, ShelfInventorySummary] = {}
        self._recent_interactions: List[ShelfInteractionEvent] = []

    def update_interactions(
        self,
        interactions: Sequence[ShelfInteractionEvent],
        timestamp_seconds: float,
    ) -> None:
        self._recent_interactions.extend(interactions)
        min_time = timestamp_seconds - self.assignment_window_seconds
        self._recent_interactions = [
            event for event in self._recent_interactions
            if event.timestamp_seconds >= min_time
        ]

    def update_inventory(
        self,
        summaries: Sequence[ShelfInventorySummary],
        frame_id: int,
        timestamp_seconds: float,
    ) -> List[InventoryChangeEvent]:
        events = []

        for summary in summaries:
            previous = self._previous_by_shelf.get(summary.shelf.id)
            self._previous_by_shelf[summary.shelf.id] = summary

            if previous is None:
                continue

            product_delta = summary.product_count - previous.product_count
            empty_delta = summary.empty_count - previous.empty_count

            event_type = None
            quantity = 0
            if product_delta < 0 or empty_delta > 0:
                event_type = "pickup"
                quantity = max(abs(product_delta), max(empty_delta, 0))
            elif product_delta > 0 or empty_delta < 0:
                event_type = "return"
                quantity = max(product_delta, abs(min(empty_delta, 0)))

            if event_type is None or quantity == 0:
                continue

            events.append(
                InventoryChangeEvent(
                    type=event_type,
                    shelf_id=summary.shelf.id,
                    frame_id=frame_id,
                    timestamp_seconds=timestamp_seconds,
                    quantity=quantity,
                    product_count_before=previous.product_count,
                    product_count_after=summary.product_count,
                    empty_count_before=previous.empty_count,
                    empty_count_after=summary.empty_count,
                    assigned_global_id=self._assign_customer(
                        summary.shelf.id,
                        timestamp_seconds,
                    ),
                )
            )

        return events

    def _assign_customer(
        self,
        shelf_id: str,
        timestamp_seconds: float,
    ) -> Optional[int]:
        candidates = [
            event for event in self._recent_interactions
            if event.shelf_id == shelf_id
            and 0 <= timestamp_seconds - event.timestamp_seconds <= self.assignment_window_seconds
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda event: event.timestamp_seconds).global_id


class SessionReportBuilder:
    def __init__(self, demographic_estimator=None) -> None:
        self.demographic_estimator = (
            demographic_estimator
            if demographic_estimator is not None
            else HeuristicDemographicEstimator()
        )
        self._unique_customers = set()
        self._track_frames = 0
        self._track_seconds_by_customer = defaultdict(float)
        self._shelf_dwell_seconds = defaultdict(float)
        self._shelf_unique_customers = defaultdict(set)
        self._demographic_observations: Dict[int, List[DemographicObservation]] = (
            defaultdict(list)
        )
        self._interaction_events: List[ShelfInteractionEvent] = []
        self._inventory_events: List[InventoryChangeEvent] = []
        self._inventory_snapshots: List[InventorySnapshot] = []
        self._last_track_seen: Dict[int, float] = {}

    def add_tracks(
        self,
        tracks: Sequence[PersonTrack],
        frame_interval_seconds: float,
        frame: np.ndarray,
    ) -> List[DemographicObservation]:
        self._track_frames += 1
        observations: List[DemographicObservation] = []
        if self.demographic_estimator is not None:
            observations = self.demographic_estimator.estimate(frame, tracks)
            for observation in observations:
                self._demographic_observations[observation.global_id].append(
                    observation
                )

        for track in tracks:
            self._unique_customers.add(track.global_id)
            self._track_seconds_by_customer[track.global_id] += frame_interval_seconds
            self._last_track_seen[track.global_id] = track.timestamp_seconds

        return observations

    def add_shelf_presence(
        self,
        tracks: Sequence[PersonTrack],
        shelves: Sequence[ShelfZone],
        frame_interval_seconds: float,
    ) -> None:
        for track in tracks:
            point = estimate_interaction_point(track.box)
            for shelf in shelves:
                if shelf.contains(point):
                    self._shelf_dwell_seconds[shelf.id] += frame_interval_seconds
                    self._shelf_unique_customers[shelf.id].add(track.global_id)

    def add_interactions(self, events: Sequence[ShelfInteractionEvent]) -> None:
        self._interaction_events.extend(events)

    def add_inventory_snapshots(
        self,
        summaries: Sequence[ShelfInventorySummary],
        frame_id: int,
        timestamp_seconds: float,
    ) -> None:
        for summary in summaries:
            self._inventory_snapshots.append(
                InventorySnapshot(
                    shelf_id=summary.shelf.id,
                    frame_id=frame_id,
                    timestamp_seconds=timestamp_seconds,
                    product_count=summary.product_count,
                    empty_count=summary.empty_count,
                    needs_restock=summary.needs_restock,
                    alert_reason=summary.alert_reason,
                )
            )

    def add_inventory_events(self, events: Sequence[InventoryChangeEvent]) -> None:
        self._inventory_events.extend(events)

    def build(self) -> Dict:
        picked_by_customer = defaultdict(int)
        returned_by_customer = defaultdict(int)
        net_purchased_by_customer = defaultdict(int)
        shelf_metrics = defaultdict(
            lambda: {
                "interactions": 0,
                "unique_customers": 0,
                "dwell_seconds": 0.0,
                "avg_dwell_seconds_per_customer": 0.0,
                "pickups": 0,
                "returns": 0,
                "picked_quantity": 0,
                "returned_quantity": 0,
                "interaction_to_pickup_rate": 0.0,
                "net_pickup_quantity": 0,
            }
        )

        for event in self._interaction_events:
            shelf_metrics[event.shelf_id]["interactions"] += 1

        for event in self._inventory_events:
            metrics = shelf_metrics[event.shelf_id]
            if event.type == "pickup":
                metrics["pickups"] += 1
                metrics["picked_quantity"] += event.quantity
                if event.assigned_global_id is not None:
                    picked_by_customer[event.assigned_global_id] += event.quantity
                    net_purchased_by_customer[event.assigned_global_id] += event.quantity
            elif event.type == "return":
                metrics["returns"] += 1
                metrics["returned_quantity"] += event.quantity
                if event.assigned_global_id is not None:
                    returned_by_customer[event.assigned_global_id] += event.quantity
                    net_purchased_by_customer[event.assigned_global_id] -= event.quantity

        for shelf_id, dwell_seconds in self._shelf_dwell_seconds.items():
            metrics = shelf_metrics[shelf_id]
            unique_customers = len(self._shelf_unique_customers[shelf_id])
            metrics["unique_customers"] = unique_customers
            metrics["dwell_seconds"] = round(dwell_seconds, 3)
            metrics["avg_dwell_seconds_per_customer"] = (
                round(dwell_seconds / unique_customers, 3)
                if unique_customers
                else 0.0
            )

        for metrics in shelf_metrics.values():
            metrics["interaction_to_pickup_rate"] = (
                round(metrics["pickups"] / metrics["interactions"], 3)
                if metrics["interactions"]
                else 0.0
            )
            metrics["net_pickup_quantity"] = (
                metrics["picked_quantity"] - metrics["returned_quantity"]
            )

        unique_count = len(self._unique_customers)
        purchased_items = sum(max(quantity, 0) for quantity in net_purchased_by_customer.values())
        demographics_by_customer = {
            global_id: summarize_demographics(observations)
            for global_id, observations in self._demographic_observations.items()
        }

        return {
            "single_visit_analytics": {
                "unique_customers": unique_count,
                "average_items_purchased_per_customer": (
                    purchased_items / unique_count if unique_count else 0.0
                ),
                "average_items_picked_per_customer": (
                    sum(picked_by_customer.values()) / unique_count
                    if unique_count
                    else 0.0
                ),
                "demographics": {
                    "status": "estimated",
                    "method": demographic_method(demographics_by_customer),
                    "apparent_age_group_distribution": distribution(
                        item["apparent_age_group"]
                        for item in demographics_by_customer.values()
                    ),
                    "gender_distribution": distribution(
                        item["gender"] for item in demographics_by_customer.values()
                    ),
                },
            },
            "customer_behavior_product_placement": {
                "shelves": dict(sorted(shelf_metrics.items())),
                "top_shelves_by_interactions": rank_shelves(
                    shelf_metrics,
                    metric="interactions",
                ),
                "top_shelves_by_conversion": rank_shelves(
                    shelf_metrics,
                    metric="interaction_to_pickup_rate",
                ),
            },
            "real_time_inventory_monitoring": {
                "latest_by_shelf": latest_inventory_by_shelf(self._inventory_snapshots),
                "restock_alerts": [
                    asdict(snapshot)
                    for snapshot in self._inventory_snapshots
                    if snapshot.needs_restock
                ],
                "timeline": [asdict(snapshot) for snapshot in self._inventory_snapshots],
            },
            "inventory_action_recognition": {
                "events": [asdict(event) for event in self._inventory_events],
                "summary": {
                    "pickup_events": sum(
                        1 for event in self._inventory_events if event.type == "pickup"
                    ),
                    "return_events": sum(
                        1 for event in self._inventory_events if event.type == "return"
                    ),
                    "picked_quantity": sum(
                        event.quantity
                        for event in self._inventory_events
                        if event.type == "pickup"
                    ),
                    "returned_quantity": sum(
                        event.quantity
                        for event in self._inventory_events
                        if event.type == "return"
                    ),
                },
            },
            "per_customer": {
                str(global_id): {
                    "observed_seconds": round(
                        self._track_seconds_by_customer[global_id],
                        3,
                    ),
                    "picked_quantity": picked_by_customer[global_id],
                    "returned_quantity": returned_by_customer[global_id],
                    "net_purchased_quantity": max(
                        net_purchased_by_customer[global_id],
                        0,
                    ),
                    "demographics": demographics_by_customer.get(
                        global_id,
                        {
                            "apparent_age_group": "unknown",
                            "gender": "unknown",
                            "confidence": 0.0,
                            "method": "none",
                        },
                    ),
                    "last_seen_seconds": round(
                        self._last_track_seen.get(global_id, 0.0),
                        3,
                    ),
                }
                for global_id in sorted(self._unique_customers)
            },
        }


def write_session_report(output_path: Path, report: Dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")


def write_inventory_timeline_csv(
    output_path: Path,
    snapshots: Sequence[Dict],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        file.write(
            "shelf_id,frame_id,timestamp_seconds,product_count,empty_count,"
            "needs_restock,alert_reason\n"
        )
        for snapshot in snapshots:
            file.write(
                f"{snapshot['shelf_id']},{snapshot['frame_id']},"
                f"{snapshot['timestamp_seconds']:.3f},"
                f"{snapshot['product_count']},{snapshot['empty_count']},"
                f"{snapshot['needs_restock']},{snapshot['alert_reason']}\n"
            )


def estimate_interaction_point(
    person_box: Tuple[float, float, float, float],
) -> Tuple[float, float]:
    x1, y1, x2, y2 = person_box
    return ((x1 + x2) / 2.0, y1 + (y2 - y1) * 0.55)


def summarize_demographics(
    observations: Sequence[DemographicObservation],
) -> Dict:
    if not observations:
        return {
            "apparent_age_group": "unknown",
            "gender": "unknown",
            "confidence": 0.0,
            "method": "none",
        }

    age_group = most_common(observation.apparent_age_group for observation in observations)
    gender = most_common(observation.gender for observation in observations)
    method = most_common(observation.method for observation in observations)
    confidences = [
        observation.confidence
        for observation in observations
        if observation.apparent_age_group == age_group
    ]
    return {
        "apparent_age_group": age_group,
        "gender": gender,
        "confidence": round(sum(confidences) / len(confidences), 3)
        if confidences
        else 0.0,
        "method": method,
    }


def demographic_method(demographics_by_customer: Dict[int, Dict]) -> str:
    methods = distribution(
        item.get("method", "none") for item in demographics_by_customer.values()
    )
    if not methods:
        return "none"
    if "buffalo_l" in methods:
        return "insightface_buffalo_l_face_age_gender"
    if "body_scale_heuristic" in methods:
        return "non_face_body_scale_heuristic"
    return ", ".join(methods)


def distribution(values) -> Dict[str, int]:
    counts = defaultdict(int)
    for value in values:
        counts[value] += 1
    return dict(sorted(counts.items()))


def most_common(values) -> str:
    counts = distribution(values)
    if not counts:
        return "unknown"
    return max(counts.items(), key=lambda item: item[1])[0]


def rank_shelves(shelf_metrics: Dict, metric: str) -> List[Dict]:
    ranked = sorted(
        shelf_metrics.items(),
        key=lambda item: item[1].get(metric, 0),
        reverse=True,
    )
    return [
        {
            "shelf_id": shelf_id,
            metric: metrics.get(metric, 0),
        }
        for shelf_id, metrics in ranked
        if metrics.get(metric, 0) > 0
    ]


def latest_inventory_by_shelf(
    snapshots: Sequence[InventorySnapshot],
) -> Dict[str, Dict]:
    latest = {}
    for snapshot in snapshots:
        latest[snapshot.shelf_id] = asdict(snapshot)
    return dict(sorted(latest.items()))
