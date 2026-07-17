# ☕ Coffeeshop Seat & Table Detector

Deteksi otomatis **orang**, **kursi**, dan **meja** dari foto/video coffeeshop menggunakan **YOLOv8** (pretrained COCO) + **Streamlit**. Dibuat sebagai fondasi untuk sistem deteksi ketersediaan kursi (seat occupancy) di coffeeshop secara real-time.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-app-red)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-orange)

## 📸 Preview

<!-- Ganti dengan screenshot hasil deteksi kamu -->
![demo](docs/demo.png)

## ✨ Fitur

- Upload foto atau video coffeeshop langsung dari browser
- Deteksi `person`, `chair`, `dining table` pakai model YOLOv8 pretrained
- Estimasi status kursi (`occupied` / `empty`) berdasarkan posisi orang di sekitar kursi
- Pilihan ukuran model (nano/small/medium) dan threshold yang bisa diatur dari sidebar
- Ringkasan jumlah orang, kursi terisi, dan kursi kosong secara otomatis

## 🚀 Quickstart

```bash
git clone https://github.com/rizqy-fadhil/NAMA-REPO.git
cd NAMA-REPO

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
streamlit run app.py
```

Buka `http://localhost:8501` di browser, upload foto/video coffeeshop, lihat hasilnya.

> Catatan: model YOLOv8 (`yolov8n.pt` dkk) akan otomatis terdownload saat pertama kali dijalankan.

## 📁 Struktur Folder

\```
seat-detector/
├── app.py
├── requirements.txt
├── utils/
│   ├── detector.py
│   └── occupancy.py
├── models/
└── sample_data/
\```

## 🧠 Cara Kerja

1. Model YOLOv8 pretrained COCO mendeteksi objek `person`, `chair`, `dining table`
2. Status kursi dihitung dari overlap (IoU) antara bounding box orang dan kursi
3. Semua parameter bisa diatur langsung dari sidebar Streamlit

## 🔧 Roadmap / Pengembangan Lanjutan

- [ ] Perbaiki logic occupancy untuk sudut kamera CCTV (bukan cuma overlap box)
- [ ] Fine-tuning model dengan dataset coffeeshop asli
- [ ] Dukungan RTSP live stream dari CCTV
- [ ] Zona kursi tetap (ROI) untuk kamera statis

## 📄 Lisensi

MIT — bebas dipakai, dimodifikasi, dan disebarluaskan.
