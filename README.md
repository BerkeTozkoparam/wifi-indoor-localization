# 📡 WiFi Fingerprinting ile Bina & Kat Tespiti

520 WiFi Access Point sinyalinden bina ve kat tahmin eden makine öğrenmesi projesi.

**[🚀 Canlı Demo](https://berketozkoparam-wifi-indoor-localization.streamlit.app)**

---

## Sonuçlar

| Metrik | Doğruluk |
|--------|----------|
| Bina Tespiti | **%98.0** |
| Kat Tespiti | **%89.6** |
| Genel | **%89.4** |

## Nasıl Çalışıyor?

```
Telefon 520 WAP sinyali ölçer
        │
        ▼
   LightGBM Modeli
   (500 karar ağacı, her biri öncekinin hatasını düzeltir)
        │
        ▼
   Tahmin: Bina X, Kat Y
```

1. **Veri**: Telefonun algıladığı WiFi sinyal güçleri (RSSI, -104 ile 0 dBm)
2. **Model**: LightGBM — Gradient Boosting tabanlı, seyrek veriyle iyi çalışır
3. **Çıktı**: 3 bina × 5 kat = 13 sınıf arasından tahmin

## Proje Yapısı

```
├── app.py                 # Streamlit interaktif web uygulaması
├── Main.py                # Model eğitimi + simülasyon (standalone script)
├── model.pkl              # Eğitilmiş LightGBM modeli
├── sample_data.csv        # Simülasyon için örnek veri
├── archive-10/
│   ├── TrainingData.csv   # 19,937 eğitim ölçümü
│   └── ValidationData.csv # 1,111 test ölçümü
└── requirements.txt
```

## Kurulum

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

[UJIndoorLoc](https://archive.ics.uci.edu/dataset/310/ujiindoorloc) — İspanya Jaume I Üniversitesi kampüsünden toplanan WiFi fingerprint verisi.

- **520** WiFi Access Point
- **3** bina, **5** kat (0-4)
- **%96.5** seyreklik (çoğu WAP sinyal algılamıyor)

## Kullanım Alanları

| Alan | Uygulama |
|------|----------|
| 🏥 Hastane | Hasta/doktor konum takibi |
| 🛒 AVM | Mağaza içi navigasyon |
| 🏭 Fabrika | Personel/ekipman takibi |
| ✈️ Havalimanı | Yolcu yönlendirme |
| 🏫 Üniversite | Kampüs navigasyonu |

## Teknolojiler

`Python` `LightGBM` `Streamlit` `Plotly` `scikit-learn` `pandas`

---

Berke Baran Tozkoparan
