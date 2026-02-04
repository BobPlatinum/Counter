# scripts/20_train.py
import sys
import re
import shutil
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------- PyQt6 (fallback PyQt5) ----------------------
try:
    from PyQt6.QtCore import Qt, QProcess
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
        QLabel, QPushButton, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
        QProgressBar, QTextEdit, QHBoxLayout, QVBoxLayout, QFormLayout,
        QGroupBox
    )
    QT_ALIGN_CENTER = Qt.AlignmentFlag.AlignCenter
except Exception:
    from PyQt5.QtCore import Qt, QProcess
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
        QLabel, QPushButton, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
        QProgressBar, QTextEdit, QHBoxLayout, QVBoxLayout, QFormLayout,
        QGroupBox
    )
    QT_ALIGN_CENTER = Qt.AlignCenter


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def find_yolo_exe() -> Path:
    """
    Prefer env-local yolo.exe to avoid PATH issues.
    Windows: <env>/Scripts/yolo.exe
    """
    py = Path(sys.executable)
    scripts = py.parent / "Scripts"
    cand = scripts / ("yolo.exe" if sys.platform.startswith("win") else "yolo")
    return cand if cand.exists() else Path("yolo")


def safe_copy(src: Path, dst: Path):
    ensure_dir(dst.parent)
    shutil.copy2(str(src), str(dst))


class TrainGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Counter - Train Tool (YOLO Segment)")
        self.resize(980, 760)

        self.yolo_exe = find_yolo_exe()
        self.proc = QProcess(self)
        self.proc.readyReadStandardOutput.connect(self.on_stdout)
        self.proc.readyReadStandardError.connect(self.on_stderr)
        self.proc.finished.connect(self.on_finished)

        self.run_dir: Path = None
        self.total_epochs = 0

        # 10 knobs (你能读懂的旋钮)
        self.data_edit = QLineEdit("coco128-seg.yaml")
        self.model_edit = QLineEdit("artifacts/weights/yolov8n-seg.pt")
        self.expname_edit = QLineEdit("seg_exp1")  # 你的实验名（也用于发布命名）

        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(1, 500); self.epochs_spin.setValue(10)
        self.imgsz_spin = QSpinBox(); self.imgsz_spin.setRange(128, 2048); self.imgsz_spin.setSingleStep(32); self.imgsz_spin.setValue(640)
        self.batch_spin = QSpinBox(); self.batch_spin.setRange(1, 256); self.batch_spin.setValue(8)
        self.device_combo = QComboBox(); self.device_combo.addItems(["0", "cpu"])
        self.workers_spin = QSpinBox(); self.workers_spin.setRange(0, 32); self.workers_spin.setValue(8)
        self.lr0_spin = QDoubleSpinBox(); self.lr0_spin.setDecimals(6); self.lr0_spin.setRange(0.0, 1.0); self.lr0_spin.setValue(0.01)
        self.patience_spin = QSpinBox(); self.patience_spin.setRange(0, 300); self.patience_spin.setValue(100)

        # fixed-ish but visible
        self.project_edit = QLineEdit("train")  # outputs to artifacts/runs/segment/train/<name>
        self.resume_edit = QLineEdit("")        # optional: path to last.pt
        self.resume_combo = QComboBox(); self.resume_combo.addItems(["False", "True"])

        # buttons
        self.btn_pick_data = QPushButton("选择 data.yaml")
        self.btn_pick_model = QPushButton("选择初始 model.pt")
        self.btn_pick_resume = QPushButton("选择 last.pt (resume)")
        self.btn_start = QPushButton("开始训练")
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)

        self.btn_pick_data.clicked.connect(self.pick_data)
        self.btn_pick_model.clicked.connect(self.pick_model)
        self.btn_pick_resume.clicked.connect(self.pick_resume_ckpt)
        self.btn_start.clicked.connect(self.start_train)
        self.btn_stop.clicked.connect(self.stop_train)

        # progress
        self.epoch_bar = QProgressBar(); self.epoch_bar.setValue(0)
        self.epoch_label = QLabel("Epoch: -"); self.epoch_label.setAlignment(QT_ALIGN_CENTER)
        self.batch_bar = QProgressBar(); self.batch_bar.setValue(0)
        self.batch_label = QLabel("Batch: -"); self.batch_label.setAlignment(QT_ALIGN_CENTER)

        # log
        self.log = QTextEdit(); self.log.setReadOnly(True)

        # layout
        root = QWidget()
        lay = QVBoxLayout(root)

        grp = QGroupBox("训练旋钮（10 个关键参数）")
        form = QFormLayout(grp)

        row_data = QWidget(); row_data_l = QHBoxLayout(row_data); row_data_l.setContentsMargins(0,0,0,0)
        row_data_l.addWidget(self.data_edit); row_data_l.addWidget(self.btn_pick_data)
        form.addRow("data", row_data)

        row_model = QWidget(); row_model_l = QHBoxLayout(row_model); row_model_l.setContentsMargins(0,0,0,0)
        row_model_l.addWidget(self.model_edit); row_model_l.addWidget(self.btn_pick_model)
        form.addRow("model", row_model)

        form.addRow("exp_name (发布名)", self.expname_edit)
        form.addRow("epochs", self.epochs_spin)
        form.addRow("imgsz", self.imgsz_spin)
        form.addRow("batch", self.batch_spin)
        form.addRow("device", self.device_combo)
        form.addRow("workers", self.workers_spin)
        form.addRow("lr0 (可选覆盖)", self.lr0_spin)
        form.addRow("patience", self.patience_spin)
        form.addRow("project (runs 子目录)", self.project_edit)

        row_resume = QWidget(); row_resume_l = QHBoxLayout(row_resume); row_resume_l.setContentsMargins(0,0,0,0)
        row_resume_l.addWidget(self.resume_combo)
        row_resume_l.addWidget(self.resume_edit)
        row_resume_l.addWidget(self.btn_pick_resume)
        form.addRow("resume", row_resume)

        lay.addWidget(grp)

        btn_row = QWidget()
        btn_row_l = QHBoxLayout(btn_row)
        btn_row_l.setContentsMargins(0,0,0,0)
        btn_row_l.addWidget(self.btn_start)
        btn_row_l.addWidget(self.btn_stop)
        lay.addWidget(btn_row)

        lay.addWidget(self.epoch_label)
        lay.addWidget(self.epoch_bar)
        lay.addWidget(self.batch_label)
        lay.addWidget(self.batch_bar)
        lay.addWidget(QLabel("训练日志"))
        lay.addWidget(self.log, stretch=1)

        self.setCentralWidget(root)

    def log_line(self, s: str):
        self.log.append(s)

    def pick_data(self):
        fn, _ = QFileDialog.getOpenFileName(self, "选择 data.yaml", str(PROJECT_ROOT), "YAML (*.yaml *.yml)")
        if fn:
            self.data_edit.setText(fn)

    def pick_model(self):
        fn, _ = QFileDialog.getOpenFileName(self, "选择模型权重(.pt)", str(PROJECT_ROOT), "PyTorch Weights (*.pt)")
        if fn:
            self.model_edit.setText(fn)

    def pick_resume_ckpt(self):
        fn, _ = QFileDialog.getOpenFileName(self, "选择 last.pt (resume)", str(PROJECT_ROOT), "PyTorch Weights (*.pt)")
        if fn:
            self.resume_edit.setText(fn)
            self.resume_combo.setCurrentText("True")

    def build_args(self) -> list:
        """
        Build Ultralytics CLI command:
          yolo segment train model=... data=... epochs=... imgsz=... batch=... device=...
        """
        data = self.data_edit.text().strip()
        model = self.model_edit.text().strip()
        exp_name = self.expname_edit.text().strip()
        project = self.project_edit.text().strip()

        epochs = int(self.epochs_spin.value())
        imgsz = int(self.imgsz_spin.value())
        batch = int(self.batch_spin.value())
        device = self.device_combo.currentText()
        workers = int(self.workers_spin.value())
        lr0 = float(self.lr0_spin.value())
        patience = int(self.patience_spin.value())

        # resume
        resume_flag = (self.resume_combo.currentText() == "True")
        resume_ckpt = self.resume_edit.text().strip()

        args = [
            "segment", "train",
            f"model={model}",
            f"data={data}",
            f"epochs={epochs}",
            f"imgsz={imgsz}",
            f"batch={batch}",
            f"device={device}",
            f"workers={workers}",
            f"patience={patience}",
            f"project={project}",
            f"name={exp_name}",
            "plots=True",
            "save=True",
        ]

        # optional override lr0 (note: if optimizer=auto it may ignore, but leaving here for learning)
        args.append(f"lr0={lr0}")

        if resume_flag:
            # important: Ultralytics expects resume=True, and model points to last.pt
            if resume_ckpt:
                args[2] = f"model={resume_ckpt}"
            args.append("resume=True")

        return args

    def start_train(self):
        try:
            self.log.clear()
            self.epoch_bar.setValue(0); self.batch_bar.setValue(0)
            self.epoch_label.setText("Epoch: -"); self.batch_label.setText("Batch: -")
            self.run_dir = None
            self.total_epochs = int(self.epochs_spin.value())

            args = self.build_args()

            self.log_line(f"[CMD] {self.yolo_exe} " + " ".join(args))
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)

            self.proc.setWorkingDirectory(str(PROJECT_ROOT))
            self.proc.start(str(self.yolo_exe), args)

            if not self.proc.waitForStarted(5000):
                raise RuntimeError("Failed to start training process. Check yolo installation.")

        except Exception as e:
            QMessageBox.critical(self, "Start Train Failed", str(e))
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)

    def stop_train(self):
        if self.proc and self.proc.state() == QProcess.ProcessState.Running:
            self.proc.kill()
            self.log_line("[STOP] process killed by user.")

    def parse_progress(self, line: str):
        # 1) capture run_dir from logs (Logging results to ...)
        m = re.search(r"Logging results to (.+)$", line)
        if m:
            p = m.group(1).strip()
            self.run_dir = Path(p)
            self.log_line(f"[INFO] run_dir detected: {self.run_dir}")

        # 2) epoch line: "1/10"
        m2 = re.search(r"^\s*(\d+)\s*/\s*(\d+)\s*$", line)
        if m2:
            # sometimes printed alone - rare
            ep = int(m2.group(1)); tot = int(m2.group(2))
            self.total_epochs = tot
            self.epoch_label.setText(f"Epoch: {ep}/{tot}")
            self.epoch_bar.setValue(int(ep * 100 / tot))

        # common epoch header row contains "Epoch" but values appear below
        m3 = re.search(r"^\s*(\d+)\s*/\s*(\d+)\s+[\d\.]+G", line)
        if m3:
            ep = int(m3.group(1)); tot = int(m3.group(2))
            self.total_epochs = tot
            self.epoch_label.setText(f"Epoch: {ep}/{tot}")
            self.epoch_bar.setValue(int(ep * 100 / tot))

        # batch progress: "... 8023/29572 ..."
        m4 = re.search(r"\s(\d+)\s*/\s*(\d+)\s", line)
        if m4 and "Epoch" not in line:
            cur = int(m4.group(1)); tot = int(m4.group(2))
            if tot > 0:
                self.batch_label.setText(f"Batch: {cur}/{tot}")
                self.batch_bar.setValue(int(cur * 100 / tot))

    def on_stdout(self):
        data = self.proc.readAllStandardOutput().data().decode(errors="ignore")
        for line in data.splitlines():
            line = line.rstrip()
            if not line:
                continue
            self.log_line(line)
            self.parse_progress(line)

    def on_stderr(self):
        data = self.proc.readAllStandardError().data().decode(errors="ignore")
        for line in data.splitlines():
            line = line.rstrip()
            if not line:
                continue
            self.log_line("[ERR] " + line)
            self.parse_progress(line)

    def publish_outputs(self):
        """
        发布逻辑（按你要求）：
        1) best.pt -> artifacts/weights/<exp_name>_best.pt
        2) counter_active.pt 更新为 best.pt
        3) artifacts/weights/published/<exp_name>_meta/ 存 best/last/args/results.*
        """
        exp_name = self.expname_edit.text().strip()
        if not exp_name:
            raise RuntimeError("exp_name is empty")

        if not self.run_dir:
            # fallback: compute expected run_dir
            project = self.project_edit.text().strip()
            self.run_dir = PROJECT_ROOT / "artifacts" / "runs" / "segment" / project / exp_name

        weights_dir = self.run_dir / "weights"
        best = weights_dir / "best.pt"
        last = weights_dir / "last.pt"
        args_yaml = self.run_dir / "args.yaml"
        results_csv = self.run_dir / "results.csv"
        results_png = self.run_dir / "results.png"

        if not best.exists():
            raise FileNotFoundError(f"best.pt not found: {best}")
        if not last.exists():
            raise FileNotFoundError(f"last.pt not found: {last}")

        out_weights = PROJECT_ROOT / "artifacts" / "weights"
        ensure_dir(out_weights)

        # (1) best publish
        published_best = out_weights / f"{exp_name}_best.pt"
        safe_copy(best, published_best)

        # (2) update active
        active = out_weights / "counter_active.pt"
        safe_copy(best, active)

        # (3) meta folder
        meta_root = out_weights / "published"
        meta_dir = meta_root / f"{exp_name}_meta"
        ensure_dir(meta_dir)

        safe_copy(best, meta_dir / "best.pt")
        safe_copy(last, meta_dir / "last.pt")
        if args_yaml.exists():
            safe_copy(args_yaml, meta_dir / "args.yaml")
        if results_csv.exists():
            safe_copy(results_csv, meta_dir / "results.csv")
        if results_png.exists():
            safe_copy(results_png, meta_dir / "results.png")

        self.log_line(f"[PUBLISH] best -> {published_best}")
        self.log_line(f"[PUBLISH] counter_active.pt updated -> {active}")
        self.log_line(f"[PUBLISH] meta -> {meta_dir}")

    def on_finished(self, exitCode: int, exitStatus):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.batch_bar.setValue(100)

        if exitCode == 0:
            self.log_line("[DONE] training finished successfully ✅")
            try:
                self.publish_outputs()
            except Exception as e:
                self.log_line("[PUBLISH FAILED] " + str(e))
                self.log_line(traceback.format_exc())
                QMessageBox.warning(self, "Publish Failed", str(e))
            else:
                QMessageBox.information(self, "Done", "训练完成并已发布 best/last/meta ✅")
        else:
            self.log_line(f"[FAIL] training exitCode={exitCode}")
            QMessageBox.warning(self, "Train Failed", f"训练异常结束 exitCode={exitCode}\n请看日志定位原因。")


def main():
    app = QApplication(sys.argv)
    w = TrainGUI()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
