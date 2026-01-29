from __future__ import annotations

from typing import Iterable
import numpy as np


def clamp_box(box, width: int, height: int):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(round(x1)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    x2 = max(0, min(int(round(x2)), width))
    y2 = max(0, min(int(round(y2)), height))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return int(x1), int(y1), int(x2), int(y2)


def books_band_roi_from_boxes(boxes: Iterable[Iterable[float]], width: int, height: int, padding_frac: float):
    boxes = list(boxes)
    if not boxes:
        return clamp_box((0, 0, width, height), width, height)
    min_y = min(box[1] for box in boxes)
    max_y = max(box[3] for box in boxes)
    pad = int(round(height * float(padding_frac)))
    return clamp_box((0, min_y - pad, width, max_y + pad), width, height)


def sort_rois_by_y(rois):
    def _y_center(box):
        x1, y1, x2, y2 = box
        return (y1 + y2) / 2.0

    return sorted(rois, key=_y_center)


def group_objects_by_shelf(objects: list[dict], threshold_px: float, threshold_frac: float):
    if not objects:
        return []
    heights = [obj.get("h", 0) for obj in objects if obj.get("h", 0) > 0]
    median_h = float(np.median(heights)) if heights else 0.0
    threshold = max(float(threshold_px), float(threshold_frac) * median_h)

    sorted_objs = sorted(objects, key=lambda o: o.get("y_center", 0))
    groups = []
    current = [sorted_objs[0]]
    for obj in sorted_objs[1:]:
        if abs(obj.get("y_center", 0) - current[-1].get("y_center", 0)) < threshold:
            current.append(obj)
        else:
            groups.append(current)
            current = [obj]
    groups.append(current)
    return groups
