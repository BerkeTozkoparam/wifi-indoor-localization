# Welding Defect Detection — FCN + YOLOv8s Ensemble Pipeline

Kaynak dikiş hatası tespiti için FCN (Fully Convolutional Network) ve YOLOv8s modellerini birleştiren bir ensemble pipeline.

## Mimari

```
Görüntü
  ├── FCN  → Anomali ısı haritası (piksel düzeyinde doku skoru)
  └── YOLO → Sınıf + Bounding Box tahmini
       ↓
  FCN prob skoru × YOLO conf → Birleşik ensemble skoru
       ↓
  Final çıktı: Sınıf + BBox + Ensemble Confidence
```

## Sınıflar

| ID | Sınıf     | Açıklama              |
|----|-----------|------------------------|
| 0  | Bad Weld  | Hatalı kaynak          |
| 1  | Good Weld | İyi kaynak             |
| 2  | Defect    | Genel kusur            |

## Ensemble Mantığı

- **Anomaly sınıfları (Bad Weld, Defect):** `ensemble = yolo_conf × (1 + α × fcn_score)` — FCN skoru yüksekse güven artar
- **Good Weld:** `ensemble = yolo_conf × (1 - α × fcn_score)` — FCN anomali bulursa ceza uygulanır
- α = 0.4 (FCN boost katsayısı)

## Kurulum

```bash
pip install -r requirements.txt
```

> Google Colab'da çalıştırmak için `Runtime → Change runtime type → A100 GPU` seçin.

## Kullanım

`welding_fcn_yolo.py` dosyası Google Colab'da adım adım çalıştırılmak üzere hazırlanmıştır:

1. **Kurulum & GPU kontrolü**
2. **Google Drive'dan dataset yükleme** (The Welding Defect Dataset v2)
3. **FCN modeli tanımı ve eğitimi** (UNet-style, Dice Loss, 50 epoch, early stopping)
4. **YOLOv8s eğitimi** (300 epoch, AdamW, cosine LR)
5. **FCN + YOLO Ensemble pipeline** (inference + görselleştirme)
6. **Sonuçları Google Drive'a kaydetme**

## Gereksinimler

- Python 3.8+
- CUDA destekli GPU (A100 önerilir)
- Google Drive'da dataset: `archive-12/The Welding Defect Dataset - v2/`

## Çıktı Formatı

Her detection için etiket formatı: `ClassName Y:<yolo_conf> E:<ensemble_score> F:<fcn_score>`
