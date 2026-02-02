print("--- Script start ---")
import sys
import argparse
from pathlib import Path

# print(sys.path[:5])
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from counter.core.pipeline import InferencePipeline

def main():
    print("--- Start main ---")
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="图片文件路径 或 文件夹路径")
    ap.add_argument("--model", default="artifacts/weights/yolov8n-seg.pt", help="分割模型权重路径")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IOU 阈值")
    ap.add_argument("--device", default="0", help="设备：0 表示 GPU0；也可写 cpu")
    ap.add_argument("--show-masks", action="store_true", help="annotated 是否叠加 mask")
    args = ap.parse_args()

    print("--- Parsed args ---")

    project_root = Path(__file__).resolve().parents[1]
    print(f"--- Project root: {project_root} ---")

    print("--- Creating pipeline ---")
    pipeline = InferencePipeline(weights_path=args.model, result_root=project_root / "result")
    print("--- Pipeline created ---")

    print("--- Running pipeline ---")
    pipeline.run(
        source=Path(args.source),
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        show_masks=args.show_masks,
    )
    print("--- End main ---")

if __name__ == "__main__":
    main()
