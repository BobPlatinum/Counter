import time
from pathlib import Path

import cv2

from counter.core.counting import extract_counts_and_detections
from counter.core.predictor import Predictor
from counter.core.render import render_annotated, render_silhouette
from counter.io.result_writer import ResultWriter
from counter.utils.files import list_images
from counter.utils.paths import safe_relpath


class InferencePipeline:
    def __init__(self, weights_path: str, result_root: Path):
        self.predictor = Predictor(weights_path)
        self.writer = ResultWriter(result_root)

    def run(self, source: Path, conf: float, iou: float, device: str, show_masks: bool = True) -> Path:
        run_dir = self.writer.new_run_dir()
        images = list_images(source)
        if not images:
            raise FileNotFoundError(f"没找到图片：{source}")

        summary = {
            "run_dir": str(run_dir),
            "model": weights_path_hint(self.predictor),
            "conf": conf,
            "iou": iou,
            "device": device,
            "images": [],
        }

        for img_path in images:
            rel = safe_relpath(img_path, source)

            # 读取原图尺寸（用于 silhouette）
            img0 = cv2.imread(str(img_path))
            h0, w0 = img0.shape[:2]

            # 推理计时
            t0 = time.perf_counter()
            r = self.predictor.predict_one(str(img_path), conf=conf, iou=iou, device=device)
            inference_ms = round((time.perf_counter() - t0) * 1000, 2)

            counts, detections = extract_counts_and_detections(r)

            annotated = render_annotated(r, counts, show_masks=show_masks)
            sil = render_silhouette(r, h0, w0)

            out_anno = run_dir / "annotated" / rel
            out_sil = run_dir / "silhouette" / rel
            self.writer.save_image(out_anno, annotated)
            self.writer.save_image(out_sil, sil)

            item = {
                "file": str(img_path),
                "annotated": str(out_anno),
                "silhouette": str(out_sil),
                "inference_ms": inference_ms,
                "counts_by_class": counts,
                "total": int(sum(counts.values())),
                "detections": detections,
            }
            summary["images"].append(item)

            print(f"[OK] {img_path.name}  total={item['total']}")

        self.writer.write_json(run_dir / "summary.json", summary)
        print(f"\nDone. Results saved to: {run_dir}")
        return run_dir


def weights_path_hint(predictor: Predictor) -> str:
    # Ultralytics YOLO 对象里有 weights 信息，但不同版本字段可能变化；先用占位说明即可
    return "see artifacts/weights"

