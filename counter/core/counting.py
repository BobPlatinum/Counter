def extract_counts_and_detections(r):
    """
    从 Ultralytics Results 中提取：
    - counts_by_class
    - detections (class_id/name/conf/bbox)
    Results/Boxes 提供 xyxy/conf/cls 等属性。
    """
    counts = {}
    detections = []

    if r.boxes is None or len(r.boxes) == 0:
        return counts, detections

    xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    clss = r.boxes.cls.cpu().numpy().astype(int)

    for i in range(len(clss)):
        cls_id = int(clss[i])
        cls_name = r.names.get(cls_id, str(cls_id))
        c = float(confs[i])
        x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]

        counts[cls_name] = counts.get(cls_name, 0) + 1
        detections.append(
            {
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": round(c, 4),
                "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            }
        )

    return counts, detections

