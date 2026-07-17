"""
detector.py
-----------
Wrapper di sekitar model YOLOv8 (pretrained COCO) untuk mendeteksi
'person', 'chair', dan 'dining table' pada gambar atau frame video.

Kelas COCO yang relevan (index bawaan COCO):
    0  -> person
    56 -> chair
    60 -> dining table

Kita tidak hardcode index-nya, tapi ambil dari model.names supaya
tetap aman kalau versi model berubah.
"""

from ultralytics import YOLO
import numpy as np
import cv2

# Kelas yang kita pakai dari 80 kelas COCO bawaan
TARGET_CLASSES = ["person", "chair", "dining table"]

# Warna box per kelas (BGR, karena kita pakai OpenCV untuk menggambar)
COLORS = {
    "person": (0, 0, 255),        # merah
    "chair": (255, 165, 0),       # oranye
    "dining table": (0, 200, 0),  # hijau
}


class SeatDetector:
    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.35):
        """
        model_path : path ke file .pt (kalau belum ada, Ultralytics akan
                     otomatis download dari internet saat pertama kali dipakai)
        conf       : confidence threshold minimum untuk sebuah deteksi ditampilkan
        """
        self.model = YOLO(model_path)
        self.conf = conf

        # Mapping id kelas -> nama, dan filter hanya id yang kita mau
        self.id_to_name = self.model.names
        self.target_ids = {
            cid: name for cid, name in self.id_to_name.items()
            if name in TARGET_CLASSES
        }

    def predict(self, image_bgr: np.ndarray):
        """
        Jalankan deteksi pada satu frame/gambar (format BGR, seperti hasil cv2.imread).
        Return: list of dict [{cls, conf, box(x1,y1,x2,y2)}, ...]
        """
        results = self.model(image_bgr, conf=self.conf, verbose=False)[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in self.target_ids:
                continue  # skip kelas yang bukan person/chair/dining table

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf_score = float(box.conf[0])
            detections.append({
                "cls": self.target_ids[cls_id],
                "conf": conf_score,
                "box": (int(x1), int(y1), int(x2), int(y2)),
            })
        return detections

    def draw(self, image_bgr: np.ndarray, detections: list) -> np.ndarray:
        """
        Gambar bounding box + label di atas image_bgr, return image baru (copy).
        """
        img = image_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cls_name = det["cls"]
            conf_score = det["conf"]
            color = COLORS.get(cls_name, (200, 200, 200))

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf_score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                img, label, (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )
        return img

    @staticmethod
    def summarize(detections: list) -> dict:
        """
        Hitung jumlah deteksi per kelas, berguna untuk ditampilkan sebagai
        metrik ringkas di Streamlit (mis. jumlah orang, kursi, meja terdeteksi).
        """
        summary = {name: 0 for name in TARGET_CLASSES}
        for det in detections:
            summary[det["cls"]] += 1
        return summary
