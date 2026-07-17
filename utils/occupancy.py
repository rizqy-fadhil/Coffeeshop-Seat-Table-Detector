"""
occupancy.py
------------
Logic sederhana untuk menentukan apakah sebuah kursi 'occupied' (terisi)
atau 'empty' (kosong), berdasarkan seberapa besar bounding box 'person'
tumpang tindih (overlap) dengan bounding box 'chair'.

Ini adalah langkah pengembangan lanjutan dari deteksi dasar (person/chair/table),
yang jadi fondasi fitur utama: "deteksi ketersediaan kursi".
"""

from typing import List, Dict


def _iou(boxA, boxB) -> float:
    """Hitung Intersection-over-Union antara dua bounding box (x1,y1,x2,y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inter_area / float(areaA + areaB - inter_area)


def get_chair_status(detections: List[Dict], overlap_thresh: float = 0.1) -> List[Dict]:
    """
    Ambil semua deteksi 'chair', lalu tandai status masing-masing:
    'occupied' kalau overlap dengan minimal satu box 'person' >= overlap_thresh,
    'empty' kalau tidak ada overlap yang cukup besar.

    overlap_thresh sengaja dibuat rendah (default 0.1) karena orang duduk di
    kursi biasanya hanya menutupi sebagian kecil area kursi dari sudut CCTV
    (bagian bawah kursi sering tertutup badan/meja), bukan menutupi semua.

    Return: list kursi, masing-masing dict {box, status}
    """
    chairs = [d for d in detections if d["cls"] == "chair"]
    people = [d for d in detections if d["cls"] == "person"]

    chair_status = []
    for chair in chairs:
        status = "empty"
        for person in people:
            if _iou(chair["box"], person["box"]) >= overlap_thresh:
                status = "occupied"
                break
        chair_status.append({"box": chair["box"], "status": status})

    return chair_status


def summarize_occupancy(chair_status: List[Dict]) -> Dict[str, int]:
    """Ringkasan jumlah kursi occupied vs empty."""
    total = len(chair_status)
    occupied = sum(1 for c in chair_status if c["status"] == "occupied")
    empty = total - occupied
    return {"total": total, "occupied": occupied, "empty": empty}
