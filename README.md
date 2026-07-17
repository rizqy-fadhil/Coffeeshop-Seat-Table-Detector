# ☕ Coffeeshop Seat & Table Detector

Aplikasi Streamlit untuk mendeteksi **orang**, **kursi**, dan **meja** dari foto/video
coffeeshop menggunakan model **YOLOv8 pretrained COCO**, sebagai fondasi untuk fitur
deteksi ketersediaan kursi (seat occupancy detection).

## 📁 Struktur Folder

```
seat-detector/
├── app.py                  # Aplikasi utama Streamlit (jalankan ini)
├── requirements.txt        # Daftar dependency
├── README.md                # Dokumentasi ini
├── models/                 # Tempat menyimpan file model .pt (opsional)
│   └── (kosong, model akan auto-download ke sini/cache Ultralytics)
├── utils/
│   ├── __init__.py
│   ├── detector.py          # Wrapper YOLOv8: load model, predict, gambar bounding box
│   └── occupancy.py         # Logic overlap person-chair -> status occupied/empty
└── sample_data/             # Taruh foto/video contoh coffeeshop kamu di sini (opsional)
```

## 🚀 Cara Menjalankan

1. Buat virtual environment (opsional tapi disarankan):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

2. Install dependency:
   ```bash
   pip install -r requirements.txt
   ```

3. Jalankan aplikasi:
   ```bash
   streamlit run app.py
   ```

4. Browser otomatis terbuka di `http://localhost:8501`.
   - Tab **Foto**: upload gambar coffeeshop, hasil deteksi + ringkasan langsung muncul.
   - Tab **Video**: upload video, klik "Mulai Proses Video", hasil deteksi diproses
     frame-per-frame dan ditampilkan live.

> Catatan: saat pertama kali dijalankan, Ultralytics akan otomatis mendownload
> file model (`yolov8n.pt` dkk) dari internet. Pastikan ada koneksi internet
> di run pertama. Setelah itu, model tersimpan di cache lokal.

## 🧠 Cara Kerja

1. **Deteksi objek** — `utils/detector.py` memuat model YOLOv8 yang sudah
   di-pretrain di dataset COCO (80 kelas umum, termasuk `person`, `chair`,
   `dining table`). Kita filter supaya hanya 3 kelas ini yang ditampilkan.

2. **Status kursi** — `utils/occupancy.py` menghitung *Intersection-over-Union*
   (IoU) antara bounding box tiap kursi dengan bounding box orang di sekitarnya.
   Kalau overlap-nya melebihi threshold tertentu, kursi dianggap `occupied`;
   kalau tidak, dianggap `empty`.

3. Semua parameter (model size, confidence threshold, overlap threshold,
   kelas yang ditampilkan) bisa diatur langsung dari sidebar Streamlit.

## 🔧 Pengembangan Lanjutan

- **Fine-tuning**: kalau akurasi kurang bagus di kondisi coffeeshop tertentu
  (pencahayaan redup, sudut kamera ekstrem, kursi tertutup meja), kumpulkan
  data foto/video dari coffeeshop asli, beri label pakai
  [Roboflow](https://roboflow.com) atau [CVAT](https://cvat.ai), lalu
  fine-tune model YOLOv8 dengan `model.train(data="dataset.yaml", ...)`.

- **Zona kursi tetap (ROI)**: untuk kamera CCTV yang posisinya statis, kamu
  bisa menandai koordinat tiap kursi secara manual sekali saja (bukan deteksi
  otomatis tiap frame), lalu cukup cek apakah ada bounding box `person` yang
  masuk ke zona tersebut. Ini lebih stabil dibanding deteksi kursi otomatis
  yang bisa berubah-ubah tiap frame karena oklusi.

- **Multi-kamera / RTSP live stream**: `detector.predict()` bisa menerima
  input dari `cv2.VideoCapture("rtsp://...")` untuk pipeline real-time,
  bukan cuma file upload.
