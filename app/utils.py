import logging
import time
from datetime import datetime
from pathlib import Path
import shutil

import cv2
import numpy as np
import torch


logger = logging.getLogger(__name__)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def pick_imgsz(quality: str, cfg: dict) -> int:
    quality = (quality or "balanced").lower()
    inf = cfg.get("inference", {})
    if quality == "fast":
        return int(inf.get("imgsz_fast", 640))
    if quality == "quality":
        return int(inf.get("imgsz_quality", 960))
    return int(inf.get("imgsz_balanced", 768))


def safe_predict(model, img, imgsz_list, device: str, **predict_kwargs):
    imgsz_unique = []
    for s in imgsz_list:
        if s not in imgsz_unique:
            imgsz_unique.append(int(s))

    device_used = device
    if device_used != "cpu" and not torch.cuda.is_available():
        device_used = "cpu"

    last_err = None
    for imgsz in imgsz_unique:
        try:
            results = model.predict(source=img, imgsz=imgsz, device=device_used, **predict_kwargs)
            return results, imgsz, device_used
        except RuntimeError as e:
            last_err = e
            msg = str(e).lower()
            if "out of memory" in msg and device_used != "cpu":
                logger.warning("CUDA OOM at imgsz=%s. Retrying...", imgsz)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise

    if device_used != "cpu":
        device_used = "cpu"
        imgsz = imgsz_unique[-1]
        results = model.predict(source=img, imgsz=imgsz, device=device_used, **predict_kwargs)
        return results, imgsz, device_used

    if last_err:
        raise last_err
    raise RuntimeError("safe_predict failed without an exception")


def save_image_rgb(image_rgb: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image_bgr)


def copy_file(src: str | Path, dst: str | Path) -> str:
    dst = Path(dst)
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return str(dst)


def ms_since(start_time: float) -> float:
    return (time.time() - start_time) * 1000.0
