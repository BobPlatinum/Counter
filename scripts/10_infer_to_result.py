import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    imgs = [p for p in source.rglob("*") if p.suffix.lower() in IMG_EXTS]
    return sorted(imgs)


def make_run_dir(result_root: Path) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = result_root / ts
    (run_dir / "annotated").mkdir(parents=True, exist_ok=True)
    (run_dir / "silhouette").mkdir(parents=True, exist_ok=True)
    return run_dir


def color_for_class(cls_id: int) -> tuple[int, int, int]:
    # 固定映射：同一类永远同一色（BGR）
    rng = np.random.default_rng(cls_id + 2026)
    bgr = rng.integers(30, 230, size=3, dtype=np.uint8)
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="图片文件路径 或 文件夹路径")
    ap.add_argument("--model", default="artifacts/weights/yolov8n-seg.pt", help="分割模型权重路径")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IOU 阈值")
    ap.add_argument("--device", default="0", help="设备：0 表示 GPU0；也可写 cpu")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    result_root = project_root / "result"
    result_root.mkdir(exist_ok=True)

    run_dir = make_run_dir(result_root)

    source = Path(args.source)
    images = list_images(source)
    if not images:
        raise SystemExit(f"没找到图片：{source}")

    # 1) 加载模型（Ultralytics Python API）:contentReference[oaicite:3]{index=3}
    model = YOLO(args.model)

    summary = {
        "run_dir": str(run_dir),
        "model": str(args.model),
        "conf": args.conf,
        "iou": args.iou,
        "device": args.device,
        "images": [],
    }

    for img_path in images:
        # 2) 推理（model.predict 支持 conf/iou/device 等参数）:contentReference[oaicite:4]{index=4}
        results = model.predict(
            source=str(img_path),
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )
        r = results[0]

        # 3) annotated 图：Results.plot() 返回 BGR numpy 图，可直接 cv2.imwrite :contentReference[oaicite:5]{index=5}
        annotated = r.plot(conf=True, labels=True, boxes=True, masks=True, color_mode="class")
        out_anno = run_dir / "annotated" / img_path.name
        cv2.imwrite(str(out_anno), annotated)

        # 4) silhouette 图：从 r.masks.data 取 mask，实心填充 :contentReference[oaicite:6]{index=6}
        img0 = cv2.imread(str(img_path))
        h0, w0 = img0.shape[:2]
        sil = np.zeros((h0, w0, 3), dtype=np.uint8)

        detections = []
        counts = {}

        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)

            masks = None
            if r.masks is not None:
                masks = r.masks.data.cpu().numpy()  # (N, H, W) :contentReference[oaicite:7]{index=7}

            for i in range(len(clss)):
                cls_id = int(clss[i])
                cls_name = r.names.get(cls_id, str(cls_id))
                c = float(confs[i])
                x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]

                counts[cls_name] = counts.get(cls_name, 0) + 1

                if masks is not None and i < masks.shape[0]:
                    m = masks[i].astype(np.uint8)  # 0/1
                    if m.shape[:2] != (h0, w0):
                        m = cv2.resize(m, (w0, h0), interpolation=cv2.INTER_NEAREST)
                    sil[m > 0] = color_for_class(cls_id)

                detections.append(
                    {
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": round(c, 4),
                        "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    }
                )

        out_sil = run_dir / "silhouette" / img_path.name
        cv2.imwrite(str(out_sil), sil)

        summary["images"].append(
            {
                "file": str(img_path),
                "annotated": str(out_anno),
                "silhouette": str(out_sil),
                "counts_by_class": counts,
                "total": int(sum(counts.values())),
                "detections": detections,
            }
        )

        print(f"[OK] {img_path.name}  total={summary['images'][-1]['total']}")

    out_json = run_dir / "summary.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDone. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
