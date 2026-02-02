import json
from datetime import datetime
from pathlib import Path

import cv2


class ResultWriter:
    def __init__(self, result_root: Path):
        self.result_root = result_root
        self.result_root.mkdir(parents=True, exist_ok=True)

    def new_run_dir(self) -> Path:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir = self.result_root / ts
        (run_dir / "annotated").mkdir(parents=True, exist_ok=True)
        (run_dir / "silhouette").mkdir(parents=True, exist_ok=True)
        return run_dir

    def save_image(self, path: Path, img):
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img)

    def write_json(self, path: Path, obj: dict):
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

