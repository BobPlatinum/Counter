# scripts/10_infer_to_result.py
import sys
import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

# --- Make project root importable when running "python scripts/xxx.py"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2

from ultralytics import YOLO  # Ultralytics API


# ---------------------- PyQt6 (fallback PyQt5) ----------------------
try:
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    from PyQt6.QtGui import QPixmap, QImage
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
        QLabel, QPushButton, QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox,
        QProgressBar, QTextEdit, QHBoxLayout, QVBoxLayout, QFormLayout,
        QGroupBox, QTabWidget, QTableWidget, QTableWidgetItem, QSplitter
    )
    QT_ALIGN_CENTER = Qt.AlignmentFlag.AlignCenter
except Exception:
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QPixmap, QImage
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
        QLabel, QPushButton, QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox,
        QProgressBar, QTextEdit, QHBoxLayout, QVBoxLayout, QFormLayout,
        QGroupBox, QTabWidget, QTableWidget, QTableWidgetItem, QSplitter
    )
    QT_ALIGN_CENTER = Qt.AlignCenter


# ---------------------- Helpers ----------------------
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def now_stamp() -> str:
    return time.strftime("%Y-%m-%d_%H%M%S", time.localtime())


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def cv_to_qpixmap(bgr: np.ndarray, max_w: int = 720, max_h: int = 540) -> QPixmap:
    """Convert OpenCV BGR image to QPixmap with scaling."""
    if bgr is None:
        return QPixmap()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    return pix.scaled(max_w, max_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)


def list_images(source: Path) -> List[Path]:
    if source.is_file():
        return [source] if source.suffix.lower() in IMAGE_EXTS else []
    imgs = []
    for p in source.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            imgs.append(p)
    return sorted(imgs)


def flatten_json(obj: Any, prefix: str = "") -> List[Tuple[str, str]]:
    """
    Flatten JSON to key/value rows: e.g. images[0].detections[1].confidence -> "0.83"
    """
    rows: List[Tuple[str, str]] = []

    def _rec(x: Any, k: str):
        if isinstance(x, dict):
            for kk, vv in x.items():
                _rec(vv, f"{k}.{kk}" if k else kk)
        elif isinstance(x, list):
            for i, vv in enumerate(x):
                _rec(vv, f"{k}[{i}]")
        else:
            rows.append((k, "" if x is None else str(x)))

    _rec(obj, prefix)
    return rows


def make_silhouette_from_masks(masks_data: Union[np.ndarray, Any], out_h: int, out_w: int) -> np.ndarray:
    """
    masks_data: torch tensor [N,H,W] or numpy
    output: BGR uint8 silhouette (white objects on black)
    """
    if masks_data is None:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # Convert torch -> numpy if needed
    if hasattr(masks_data, "detach"):
        masks_np = masks_data.detach().cpu().numpy()
    else:
        masks_np = np.asarray(masks_data)

    if masks_np.ndim != 3 or masks_np.shape[0] == 0:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # union of instances
    union = (masks_np > 0.5).any(axis=0).astype(np.uint8) * 255
    sil = np.stack([union, union, union], axis=-1)
    # ensure size
    sil = cv2.resize(sil, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return sil


# ---------------------- Inference core ----------------------
def run_inference_to_run_dir(
    model_path: Path,
    source: Path,
    dataset_bucket: str,   # "coco" | "coco128" | "other"
    conf: float,
    iou: float,
    device: str,
    imgsz: int,
    max_det: int,
    classes: str,          # "" or "0,1,2"
    verbose_log_cb=None,
    progress_cb=None,
) -> Tuple[Path, Path]:
    """
    Create run_dir:
      artifacts/runs/segment/{bucket}_predict/{timestamp}/
        annotated/
        silhouette/
        summary.json
    """
    # bucket dir (avoid spaces for Windows)
    bucket_dirname = f"{dataset_bucket}_predict"
    run_dir = PROJECT_ROOT / "artifacts" / "runs" / "segment" / bucket_dirname / now_stamp()
    ann_dir = run_dir / "annotated"
    sil_dir = run_dir / "silhouette"
    ensure_dir(ann_dir)
    ensure_dir(sil_dir)

    if verbose_log_cb:
        verbose_log_cb(f"[RUN] run_dir = {run_dir}")

    yolo = YOLO(str(model_path))

    img_paths = list_images(source)
    if not img_paths:
        raise RuntimeError(f"No images found in source: {source}")

    # parse classes filter
    class_list = None
    if classes.strip():
        class_list = [int(x.strip()) for x in classes.split(",") if x.strip().isdigit()]

    summary: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "model": str(model_path).replace(str(PROJECT_ROOT) + "\\", "").replace(str(PROJECT_ROOT) + "/", ""),
        "conf": conf,
        "iou": iou,
        "device": device,
        "imgsz": imgsz,
        "max_det": max_det,
        "classes": class_list,
        "bucket": dataset_bucket,
        "source": str(source),
        "images": []
    }

    total = len(img_paths)
    for idx, img_path in enumerate(img_paths, start=1):
        if progress_cb:
            progress_cb(idx - 1, total, img_path.name)

        t0 = time.time()
        preds = yolo.predict(
            source=str(img_path),
            conf=conf,
            iou=iou,
            device=device,
            imgsz=imgsz,
            max_det=max_det,
            classes=class_list,
            verbose=False
        )
        infer_ms = (time.time() - t0) * 1000.0

        r0 = preds[0]
        # annotated
        ann = r0.plot()  # numpy image (BGR) with labels/conf  :contentReference[oaicite:2]{index=2}
        ann_path = ann_dir / img_path.name
        cv2.imwrite(str(ann_path), ann)

        # silhouette
        h, w = ann.shape[:2]
        masks_data = None
        if getattr(r0, "masks", None) is not None and getattr(r0.masks, "data", None) is not None:
            masks_data = r0.masks.data  # tensor [N,H,W]  :contentReference[oaicite:3]{index=3}
        sil = make_silhouette_from_masks(masks_data, out_h=h, out_w=w)
        sil_path = sil_dir / img_path.name
        cv2.imwrite(str(sil_path), sil)

        # boxes
        counts: Dict[str, int] = {}
        detections: List[Dict[str, Any]] = []

        if getattr(r0, "boxes", None) is not None and len(r0.boxes) > 0:
            xyxy = r0.boxes.xyxy
            confs = r0.boxes.conf
            clss = r0.boxes.cls

            # torch -> numpy
            if hasattr(xyxy, "detach"):
                xyxy = xyxy.detach().cpu().numpy()
                confs = confs.detach().cpu().numpy()
                clss = clss.detach().cpu().numpy()
            else:
                xyxy = np.asarray(xyxy)
                confs = np.asarray(confs)
                clss = np.asarray(clss)

            for bb, cf, ci in zip(xyxy, confs, clss):
                class_id = int(ci)
                class_name = yolo.names.get(class_id, str(class_id))
                counts[class_name] = counts.get(class_name, 0) + 1
                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": float(cf),
                    "bbox_xyxy": [float(x) for x in bb.tolist()]
                })

        total_count = int(sum(counts.values()))

        summary["images"].append({
            "file": str(img_path),
            "annotated": str(ann_path),
            "silhouette": str(sil_path),
            "inference_ms": round(float(infer_ms), 2),
            "counts_by_class": counts,
            "total": total_count,
            "detections": detections
        })

        if verbose_log_cb:
            verbose_log_cb(f"[OK] {img_path.name}  total={total_count}  infer_ms={infer_ms:.2f}")

        if progress_cb:
            progress_cb(idx, total, img_path.name)

    summary_path = run_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if verbose_log_cb:
        verbose_log_cb(f"[DONE] summary.json -> {summary_path}")

    return run_dir, summary_path


# ---------------------- Worker Thread ----------------------
class InferenceWorker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int, int, str)   # current, total, filename
    finished_ok = pyqtSignal(str, str)     # run_dir, summary_path
    failed = pyqtSignal(str)

    def __init__(
        self,
        model_path: Path,
        source: Path,
        bucket: str,
        conf: float,
        iou: float,
        device: str,
        imgsz: int,
        max_det: int,
        classes: str,
    ):
        super().__init__()
        self.model_path = model_path
        self.source = source
        self.bucket = bucket
        self.conf = conf
        self.iou = iou
        self.device = device
        self.imgsz = imgsz
        self.max_det = max_det
        self.classes = classes

    def run(self):
        try:
            def _log(s: str):
                self.log.emit(s)

            def _prog(cur: int, tot: int, fn: str):
                self.progress.emit(cur, tot, fn)

            run_dir, summary_path = run_inference_to_run_dir(
                model_path=self.model_path,
                source=self.source,
                dataset_bucket=self.bucket,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                imgsz=self.imgsz,
                max_det=self.max_det,
                classes=self.classes,
                verbose_log_cb=_log,
                progress_cb=_prog
            )
            self.finished_ok.emit(str(run_dir), str(summary_path))
        except Exception:
            self.failed.emit(traceback.format_exc())


# ---------------------- GUI ----------------------
class InferGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Counter - Inference Tool (YOLO Segment)")
        self.resize(1280, 820)

        self.current_source: Path = PROJECT_ROOT / "assets" / "examples"
        self.current_model: Path = PROJECT_ROOT / "artifacts" / "weights" / "counter_active.pt"
        self.current_summary_path: Path = None

        # ---- Controls ----
        self.model_edit = QLineEdit(str(self.current_model))
        self.btn_pick_model = QPushButton("选择模型(.pt)")
        self.btn_pick_model.clicked.connect(self.pick_model)

        self.source_edit = QLineEdit(str(self.current_source))
        self.btn_pick_file = QPushButton("选择图片")
        self.btn_pick_dir = QPushButton("选择文件夹")
        self.btn_pick_file.clicked.connect(self.pick_source_file)
        self.btn_pick_dir.clicked.connect(self.pick_source_dir)

        self.bucket_combo = QComboBox()
        self.bucket_combo.addItems(["auto", "coco", "coco128", "other"])

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.0, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.7)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["0", "cpu"])

        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(128, 2048)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)

        self.maxdet_spin = QSpinBox()
        self.maxdet_spin.setRange(1, 3000)
        self.maxdet_spin.setValue(300)

        self.classes_edit = QLineEdit("")
        self.classes_edit.setPlaceholderText("可选：类别过滤，例如 0,5,36（留空=全部）")

        self.btn_run = QPushButton("开始推理")
        self.btn_run.clicked.connect(self.start_infer)

        # ---- Progress ----
        self.prog = QProgressBar()
        self.prog.setValue(0)
        self.prog_label = QLabel("进度：-")
        self.prog_label.setAlignment(QT_ALIGN_CENTER)

        # ---- Log ----
        self.log = QTextEdit()
        self.log.setReadOnly(True)

        # ---- Result preview ----
        self.preview_annot = QLabel("Annotated Preview")
        self.preview_annot.setAlignment(QT_ALIGN_CENTER)
        self.preview_sil = QLabel("Silhouette Preview")
        self.preview_sil.setAlignment(QT_ALIGN_CENTER)

        # ---- Summary tables ----
        self.btn_load_summary = QPushButton("载入 summary.json")
        self.btn_load_summary.clicked.connect(self.load_summary)
        self.btn_apply_params = QPushButton("从 summary 套用参数到旋钮")
        self.btn_apply_params.clicked.connect(self.apply_params_from_summary)

        self.table_flat = QTableWidget(0, 2)
        self.table_flat.setHorizontalHeaderLabels(["Key", "Value"])
        self.table_images = QTableWidget(0, 4)
        self.table_images.setHorizontalHeaderLabels(["Image", "Total", "inference_ms", "Path"])
        self.table_images.cellClicked.connect(self.on_image_row_clicked)

        self.table_det = QTableWidget(0, 4)
        self.table_det.setHorizontalHeaderLabels(["class", "conf", "x1,y1", "x2,y2"])

        # ---- Layout ----
        left = QWidget()
        left_layout = QVBoxLayout(left)

        grp = QGroupBox("推理参数（旋钮）")
        form = QFormLayout(grp)

        # model row
        model_row = QWidget()
        model_row_l = QHBoxLayout(model_row)
        model_row_l.setContentsMargins(0, 0, 0, 0)
        model_row_l.addWidget(self.model_edit)
        model_row_l.addWidget(self.btn_pick_model)
        form.addRow("model", model_row)

        # source row
        source_row = QWidget()
        source_row_l = QHBoxLayout(source_row)
        source_row_l.setContentsMargins(0, 0, 0, 0)
        source_row_l.addWidget(self.source_edit)
        source_row_l.addWidget(self.btn_pick_file)
        source_row_l.addWidget(self.btn_pick_dir)
        form.addRow("source", source_row)

        form.addRow("dataset bucket", self.bucket_combo)
        form.addRow("conf", self.conf_spin)
        form.addRow("iou", self.iou_spin)
        form.addRow("device", self.device_combo)
        form.addRow("imgsz", self.imgsz_spin)
        form.addRow("max_det", self.maxdet_spin)
        form.addRow("classes", self.classes_edit)

        left_layout.addWidget(grp)
        left_layout.addWidget(self.btn_run)
        left_layout.addWidget(self.prog_label)
        left_layout.addWidget(self.prog)
        left_layout.addWidget(QLabel("日志"))
        left_layout.addWidget(self.log, stretch=1)

        # right: tabs
        tabs = QTabWidget()

        # tab 1: preview
        tab_preview = QWidget()
        t1 = QHBoxLayout(tab_preview)
        t1.addWidget(self.preview_annot, 1)
        t1.addWidget(self.preview_sil, 1)
        tabs.addTab(tab_preview, "结果预览")

        # tab 2: summary
        tab_sum = QWidget()
        t2 = QVBoxLayout(tab_sum)

        btn_row = QWidget()
        btn_row_l = QHBoxLayout(btn_row)
        btn_row_l.setContentsMargins(0, 0, 0, 0)
        btn_row_l.addWidget(self.btn_load_summary)
        btn_row_l.addWidget(self.btn_apply_params)
        t2.addWidget(btn_row)

        split = QSplitter()
        split.setOrientation(Qt.Orientation.Vertical)
        w_top = QWidget()
        w_top_l = QVBoxLayout(w_top)
        w_top_l.addWidget(QLabel("images[] 表格（点击一行会更新预览与 detections 表）"))
        w_top_l.addWidget(self.table_images)
        w_mid = QWidget()
        w_mid_l = QVBoxLayout(w_mid)
        w_mid_l.addWidget(QLabel("detections 表"))
        w_mid_l.addWidget(self.table_det)
        w_bot = QWidget()
        w_bot_l = QVBoxLayout(w_bot)
        w_bot_l.addWidget(QLabel("summary.json 全量扁平化表格（“所有参数”都在这里）"))
        w_bot_l.addWidget(self.table_flat)

        split.addWidget(w_top)
        split.addWidget(w_mid)
        split.addWidget(w_bot)
        split.setSizes([220, 220, 320])

        t2.addWidget(split)
        tabs.addTab(tab_sum, "summary.json 表格")

        right = tabs

        main_split = QSplitter()
        main_split.addWidget(left)
        main_split.addWidget(right)
        main_split.setSizes([420, 860])

        container = QWidget()
        lay = QHBoxLayout(container)
        lay.addWidget(main_split)
        self.setCentralWidget(container)

        self.worker: InferenceWorker = None

    # ----- UI Actions -----
    def log_line(self, s: str):
        self.log.append(s)

    def pick_model(self):
        fn, _ = QFileDialog.getOpenFileName(self, "选择模型权重", str(PROJECT_ROOT), "PyTorch Weights (*.pt)")
        if fn:
            self.model_edit.setText(fn)

    def pick_source_file(self):
        fn, _ = QFileDialog.getOpenFileName(self, "选择图片", str(PROJECT_ROOT), "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)")
        if fn:
            self.source_edit.setText(fn)

    def pick_source_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择文件夹", str(PROJECT_ROOT))
        if d:
            self.source_edit.setText(d)

    def infer_bucket_auto(self, model_path: Path) -> str:
        s = str(model_path).lower()
        if "coco128" in s:
            return "coco128"
        if "coco" in s:
            return "coco"
        return "other"

    def start_infer(self):
        try:
            model_path = Path(self.model_edit.text().strip())
            source = Path(self.source_edit.text().strip())

            if not model_path.exists():
                raise FileNotFoundError(f"model not found: {model_path}")
            if not source.exists():
                raise FileNotFoundError(f"source not found: {source}")

            bucket = self.bucket_combo.currentText()
            if bucket == "auto":
                bucket = self.infer_bucket_auto(model_path)

            conf = float(self.conf_spin.value())
            iou = float(self.iou_spin.value())
            device = self.device_combo.currentText()
            imgsz = int(self.imgsz_spin.value())
            max_det = int(self.maxdet_spin.value())
            classes = self.classes_edit.text().strip()

            self.log.clear()
            self.preview_annot.setText("Running...")
            self.preview_sil.setText("Running...")
            self.prog.setValue(0)
            self.prog_label.setText("进度：准备中...")

            self.worker = InferenceWorker(
                model_path=model_path,
                source=source,
                bucket=bucket,
                conf=conf,
                iou=iou,
                device=device,
                imgsz=imgsz,
                max_det=max_det,
                classes=classes
            )
            self.worker.log.connect(self.log_line)
            self.worker.progress.connect(self.on_progress)
            self.worker.finished_ok.connect(self.on_finished)
            self.worker.failed.connect(self.on_failed)
            self.btn_run.setEnabled(False)
            self.worker.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def on_progress(self, cur: int, tot: int, fn: str):
        if tot <= 0:
            return
        pct = int(cur * 100 / tot)
        self.prog.setValue(min(max(pct, 0), 100))
        self.prog_label.setText(f"进度：{cur}/{tot}  当前：{fn}")

    def on_finished(self, run_dir: str, summary_path: str):
        self.btn_run.setEnabled(True)
        self.prog.setValue(100)
        self.prog_label.setText(f"完成 ✅ 结果目录：{run_dir}")
        self.log_line(f"[GUI] run_dir = {run_dir}")
        self.log_line(f"[GUI] summary.json = {summary_path}")
        self.current_summary_path = Path(summary_path)
        # auto-load summary to tables
        self.load_summary_from_path(self.current_summary_path)

    def on_failed(self, tb: str):
        self.btn_run.setEnabled(True)
        QMessageBox.critical(self, "Inference Failed", tb)

    # ----- Summary UI -----
    def load_summary(self):
        fn, _ = QFileDialog.getOpenFileName(self, "选择 summary.json", str(PROJECT_ROOT), "JSON (*.json)")
        if fn:
            self.load_summary_from_path(Path(fn))

    def load_summary_from_path(self, p: Path):
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            self.current_summary_path = p
            self.fill_tables_from_summary(obj)
            self.log_line(f"[LOAD] summary loaded: {p}")
        except Exception as e:
            QMessageBox.critical(self, "Load summary.json Failed", str(e))

    def fill_tables_from_summary(self, obj: Dict[str, Any]):
        # 1) flat table: all params
        flat = flatten_json(obj)
        self.table_flat.setRowCount(len(flat))
        for i, (k, v) in enumerate(flat):
            self.table_flat.setItem(i, 0, QTableWidgetItem(k))
            self.table_flat.setItem(i, 1, QTableWidgetItem(v))

        # 2) images table
        imgs = obj.get("images", [])
        self.table_images.setRowCount(len(imgs))
        for i, it in enumerate(imgs):
            self.table_images.setItem(i, 0, QTableWidgetItem(Path(it.get("file", "")).name))
            self.table_images.setItem(i, 1, QTableWidgetItem(str(it.get("total", ""))))
            self.table_images.setItem(i, 2, QTableWidgetItem(str(it.get("inference_ms", ""))))
            self.table_images.setItem(i, 3, QTableWidgetItem(str(it.get("annotated", ""))))

        # default: show first image
        if imgs:
            self.show_image_from_summary(obj, 0)

    def on_image_row_clicked(self, row: int, col: int):
        if not self.current_summary_path:
            return
        with open(self.current_summary_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        self.show_image_from_summary(obj, row)

    def show_image_from_summary(self, obj: Dict[str, Any], idx: int):
        imgs = obj.get("images", [])
        if idx < 0 or idx >= len(imgs):
            return
        it = imgs[idx]

        ann_path = Path(it.get("annotated", ""))
        sil_path = Path(it.get("silhouette", ""))

        ann = cv2.imread(str(ann_path)) if ann_path.exists() else None
        sil = cv2.imread(str(sil_path)) if sil_path.exists() else None

        if ann is not None:
            self.preview_annot.setPixmap(cv_to_qpixmap(ann))
        else:
            self.preview_annot.setText(f"annotated not found:\n{ann_path}")

        if sil is not None:
            self.preview_sil.setPixmap(cv_to_qpixmap(sil))
        else:
            self.preview_sil.setText(f"silhouette not found:\n{sil_path}")

        dets = it.get("detections", [])
        self.table_det.setRowCount(len(dets))
        for i, d in enumerate(dets):
            self.table_det.setItem(i, 0, QTableWidgetItem(str(d.get("class_name", ""))))
            self.table_det.setItem(i, 1, QTableWidgetItem(f"{float(d.get('confidence', 0.0)):.4f}"))
            bb = d.get("bbox_xyxy", [0, 0, 0, 0])
            self.table_det.setItem(i, 2, QTableWidgetItem(f"{bb[0]:.1f},{bb[1]:.1f}"))
            self.table_det.setItem(i, 3, QTableWidgetItem(f"{bb[2]:.1f},{bb[3]:.1f}"))

    def apply_params_from_summary(self):
        """
        把 summary.json 的关键参数套回旋钮（你就不用“背命令”了，直接用历史实验配置）
        """
        if not self.current_summary_path:
            QMessageBox.information(self, "提示", "请先载入一个 summary.json")
            return
        with open(self.current_summary_path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        # try apply
        if "model" in obj:
            # model could be relative path
            mp = obj["model"]
            mp_path = (PROJECT_ROOT / mp) if not Path(mp).is_absolute() else Path(mp)
            if mp_path.exists():
                self.model_edit.setText(str(mp_path))
        if "conf" in obj:
            self.conf_spin.setValue(float(obj["conf"]))
        if "iou" in obj:
            self.iou_spin.setValue(float(obj["iou"]))
        if "device" in obj:
            dv = str(obj["device"])
            # accept "0" or "cpu"
            idx = self.device_combo.findText(dv)
            if idx >= 0:
                self.device_combo.setCurrentIndex(idx)
        if "imgsz" in obj:
            self.imgsz_spin.setValue(int(obj["imgsz"]))
        if "max_det" in obj:
            self.maxdet_spin.setValue(int(obj["max_det"]))
        if "bucket" in obj:
            b = str(obj["bucket"])
            idx = self.bucket_combo.findText(b)
            if idx >= 0:
                self.bucket_combo.setCurrentIndex(idx)

        # classes
        cls = obj.get("classes", None)
        if isinstance(cls, list) and cls:
            self.classes_edit.setText(",".join([str(x) for x in cls]))
        else:
            self.classes_edit.setText("")

        self.log_line("[APPLY] 已从 summary.json 套用关键参数到旋钮 ✅")


def main():
    app = QApplication(sys.argv)
    w = InferGUI()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
