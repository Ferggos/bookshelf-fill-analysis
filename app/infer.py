import logging
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .config import repo_root
from .metrics import (
    books_band_roi_from_boxes,
    clamp_box,
    group_objects_by_shelf,
)
from .utils import pick_imgsz, safe_predict, ms_since


logger = logging.getLogger(__name__)

_MODELS: dict = {}


def _resolve_model_path(path_str: str) -> str:
    root = repo_root()
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    candidate = root / path
    if candidate.exists():
        return str(candidate)
    return path_str


def _load_model(path_str: str | None, required: bool):
    if not path_str:
        if required:
            raise ValueError("Model path is required")
        return None
    model_path = _resolve_model_path(path_str)
    try:
        return YOLO(model_path)
    except Exception as e:  # noqa: BLE001
        if required:
            raise
        logger.warning("Could not load model %s: %s", model_path, e)
        return None


def get_models(cfg: dict) -> dict:
    if _MODELS:
        return _MODELS
    paths = cfg.get("paths", {})
    _MODELS["seg"] = _load_model(paths.get("seg_model"), required=True)
    _MODELS["cls"] = _load_model(paths.get("cls_model"), required=False)
    return _MODELS


def _extract_book_objects(seg_result, cfg: dict, img_shape):
    objects = []
    if seg_result is None or seg_result.masks is None:
        return objects, 0, 0

    names = seg_result.names or {}
    target_name = cfg.get("names", {}).get("seg_class_name", "")
    name_values = list(names.values())
    use_all = target_name not in name_values

    img_h, img_w = img_shape[:2]
    img_area = float(img_h * img_w)
    inf_cfg = cfg.get("inference", {})
    use_filters = bool(inf_cfg.get("noise_filters_default", True))
    min_area_frac = float(inf_cfg.get("min_mask_area_frac", 0.0)) if use_filters else 0.0
    min_bbox_w = float(inf_cfg.get("min_bbox_w_px", 0.0)) if use_filters else 0.0

    masks_data = seg_result.masks.data
    polygons = seg_result.masks.xy

    count_before = 0
    for i, poly in enumerate(polygons):
        cls_name = ""
        if seg_result.boxes is not None and seg_result.boxes.cls is not None:
            cls_id = int(seg_result.boxes.cls[i])
            cls_name = names.get(cls_id, "")
        if not use_all and cls_name != target_name:
            continue

        count_before += 1
        if poly is None or len(poly) < 3:
            continue

        if seg_result.boxes is not None:
            box = seg_result.boxes.xyxy[i].cpu().numpy().tolist()
            conf = None
            if seg_result.boxes.conf is not None:
                conf = float(seg_result.boxes.conf[i].item())
        else:
            xs = poly[:, 0]
            ys = poly[:, 1]
            box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
            conf = None

        box = clamp_box(box, img_w, img_h)
        if (box[2] - box[0]) < min_bbox_w:
            continue

        if masks_data is not None:
            mask_area = float(masks_data[i].sum())
            if mask_area < min_area_frac * img_area:
                continue

        obj = {
            "polygon": np.array(poly, dtype=np.float32),
            "bbox": box,
            "conf": conf,
            "y_center": (box[1] + box[3]) / 2.0,
            "h": box[3] - box[1],
        }
        objects.append(obj)

    return objects, count_before, len(objects)


def _draw_objects(image_bgr, objects, mask_color=(255, 0, 0), alpha: float = 0.35):
    overlay = image_bgr.copy()
    height, width = image_bgr.shape[:2]
    for obj in objects:
        poly = obj.get("polygon")
        if poly is None or len(poly) < 3:
            continue
        pts = np.round(poly).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
        cv2.fillPoly(overlay, [pts], mask_color)

    blended = cv2.addWeighted(overlay, alpha, image_bgr, 1.0 - alpha, 0)

    for obj in objects:
        x1, y1, x2, y2 = map(int, obj.get("bbox", (0, 0, 0, 0)))
        cv2.rectangle(blended, (x1, y1), (x2, y2), (255, 255, 0), 1)
        conf = obj.get("conf")
        if conf is not None:
            label = f"{conf:.2f}"
            cv2.putText(
                blended,
                label,
                (x1, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

    return blended


def _draw_rois(image_bgr, per_shelf, overall):
    overlay = image_bgr.copy()
    multi = len(per_shelf) > 1
    for idx, shelf in enumerate(per_shelf, start=1):
        x1, y1, x2, y2 = shelf.get("roi_bbox") or shelf.get("roi_box", [0, 0, 0, 0])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 0), 2)
        prefix = "Shelf" if multi else "Band"
        label = f"{prefix} {idx}: books={shelf.get('book_count', 0)}"
        cv2.putText(
            overlay,
            label,
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 0),
            1,
            cv2.LINE_AA,
        )

    text = (
        f"Overall: shelves={overall.get('shelf_count', len(per_shelf))} "
        f"| books={overall.get('total_books', 0)}"
    )
    x_text = max(10, image_bgr.shape[1] - 420)
    cv2.putText(
        overlay,
        text,
        (x_text, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


def _build_rois_and_groups(objects, roi_mode: str, img_shape, cfg: dict):
    img_h, img_w = img_shape[:2]
    roi_mode = (roi_mode or cfg.get("inference", {}).get("roi_mode_default", "books_band")).lower()
    inf_cfg = cfg.get("inference", {})
    padding_frac = float(inf_cfg.get("roi_band_padding", 0.05))
    threshold_px = float(inf_cfg.get("per_shelf_threshold_px", 25))
    threshold_frac = float(inf_cfg.get("per_shelf_threshold_frac", 0.6))

    if roi_mode == "books_band":
        boxes = [obj["bbox"] for obj in objects]
        roi = books_band_roi_from_boxes(boxes, img_w, img_h, padding_frac)
        return [roi], [objects], roi_mode

    if roi_mode == "per_shelf":
        groups = group_objects_by_shelf(objects, threshold_px, threshold_frac)
        if not groups:
            return [clamp_box((0, 0, img_w, img_h), img_w, img_h)], [[]], roi_mode
        rois = []
        for group in groups:
            boxes = [obj["bbox"] for obj in group]
            rois.append(books_band_roi_from_boxes(boxes, img_w, img_h, padding_frac))
        return rois, groups, roi_mode

    boxes = [obj["bbox"] for obj in objects]
    roi = books_band_roi_from_boxes(boxes, img_w, img_h, padding_frac)
    return [roi], [objects], "books_band"


def run_image_inference(
    image_rgb: np.ndarray,
    cfg: dict,
    roi_mode: str,
    quality: str,
    conf: float,
    iou: float,
    run_classification: bool,
    save_dir: str | Path | None = None,
    output_name: str | None = None,
):
    start = time.time()
    metrics: dict = {"error": None}
    errors = []

    if image_rgb is None:
        metrics["error"] = "No image provided"
        return None, metrics, None

    models = get_models(cfg)
    seg_model = models.get("seg")
    cls_model = models.get("cls")

    inf_cfg = cfg.get("inference", {})
    imgsz_pref = pick_imgsz(quality, cfg)
    imgsz_fast = int(inf_cfg.get("imgsz_fast", 640))
    imgsz_list = [imgsz_pref, imgsz_fast]

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    try:
        seg_results, imgsz_used, device_used = safe_predict(
            seg_model,
            image_bgr,
            imgsz_list,
            device="cuda",
            conf=conf,
            iou=iou,
            retina_masks=bool(inf_cfg.get("retina_masks", False)),
            verbose=False,
        )
        seg_result = seg_results[0]
    except Exception as e:  # noqa: BLE001
        metrics["error"] = f"Segmentation failed: {e}"
        annotated = image_bgr.copy()
        metrics["processing_ms"] = round(ms_since(start), 2)
        metrics["imgsz_used"] = None
        metrics["device_used"] = None
        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), metrics, None

    objects, count_before, count_after = _extract_book_objects(seg_result, cfg, image_bgr.shape)
    roi_mode_selected = (roi_mode or inf_cfg.get("roi_mode_default", "books_band")).lower()
    rois, groups, roi_mode_used = _build_rois_and_groups(objects, roi_mode_selected, image_bgr.shape, cfg)

    shelves = []
    total_books = 0
    for idx, (roi, group) in enumerate(zip(rois, groups), start=1):
        book_count = int(len(group))
        total_books += book_count
        shelves.append(
            {
                "shelf_index": idx,
                "roi_bbox": list(map(int, roi)),
                "book_count": book_count,
            }
        )

    processing_ms = round(ms_since(start), 2)
    overall = {
        "shelf_count": len(shelves),
        "total_books": total_books,
        "imgsz_used": imgsz_used,
        "device_used": device_used,
        "processing_ms": processing_ms,
        "conf": conf,
        "iou": iou,
    }

    cls_info = None
    if run_classification and cls_model is not None:
        try:
            cls_results, _, _ = safe_predict(
                cls_model,
                image_bgr,
                [224],
                device=device_used,
                verbose=False,
            )
            cls_result = cls_results[0]
            if cls_result.probs is not None:
                top1 = int(cls_result.probs.top1)
                cls_info = {
                    "top1": cls_result.names.get(top1, str(top1)),
                    "conf": float(cls_result.probs.top1conf),
                    "top5": [cls_result.names.get(i, str(i)) for i in cls_result.probs.top5],
                }
        except Exception as e:  # noqa: BLE001
            cls_info = {"error": f"Classification failed: {e}"}
            errors.append(f"Classification failed: {e}")

    annotated = _draw_objects(image_bgr, objects)
    annotated = _draw_rois(annotated, shelves, overall)

    roi_bbox_top = shelves[0].get("roi_bbox") if len(shelves) == 1 else None

    metrics.update(
        {
            "shelves": shelves,
            "overall": overall,
            "cls": cls_info,
            "roi_bbox": roi_bbox_top,
        }
    )
    if errors:
        metrics["error"] = "; ".join(errors)

    output_path = None
    if save_dir and output_name:
        output_path = str(Path(save_dir) / output_name)
        cv2.imwrite(output_path, annotated)

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), metrics, output_path
