import cv2
import numpy as np
from ultralytics.utils.plotting import Colors

# Ultralytics 官方调色板工具类 Colors。
COLORS = Colors()


def color_for_class(cls_id: int) -> tuple[int, int, int]:
    b, g, r = COLORS(cls_id, True)  # True => BGR
    return int(b), int(g), int(r)


def render_annotated(r, counts: dict, show_masks: bool = True):
    """
    Results.plot() 生成可视化图（框/标签/置信度/mask）。
    再叠加 total + counts_by_class。
    """
    annotated = r.plot(conf=True, labels=True, boxes=True, masks=show_masks, color_mode="class")

    total = int(sum(counts.values()))
    parts = [f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    text = f"total:{total}  " + ("  ".join(parts) if parts else "no detections")

    cv2.putText(annotated, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(annotated, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated


def render_silhouette(r, orig_h: int, orig_w: int):
    """
    用 r.masks.data 生成实例 mask 的实心剪影图。
    """
    sil = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

    if r.masks is None or r.boxes is None or len(r.boxes) == 0:
        return sil

    clss = r.boxes.cls.cpu().numpy().astype(int)
    masks = r.masks.data.cpu().numpy()  # (N, H, W)

    for i in range(min(len(clss), masks.shape[0])):
        cls_id = int(clss[i])
        m = masks[i].astype(np.uint8)
        if m.shape[:2] != (orig_h, orig_w):
            m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        sil[m > 0] = color_for_class(cls_id)

    return sil

