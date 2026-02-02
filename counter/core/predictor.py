from ultralytics import YOLO


class Predictor:
    """
    只负责：加载模型 + 对单张图片做 predict。
    Ultralytics 提供 YOLO Python API：YOLO(weights) + model.predict(...)。
    """

    def __init__(self, weights_path: str):
        self.model = YOLO(weights_path)

    def predict_one(self, image_path: str, conf: float, iou: float, device: str):
        results = self.model.predict(
            source=image_path,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )
        return results[0]  # 单张图对应一个 Results

