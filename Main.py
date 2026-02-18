"""
WiFi Fingerprinting ile Bina ve Kat Tespiti
============================================
Model: LightGBM (Light Gradient Boosting Machine)

MANTIK:
-------
1. Her WiFi Access Point (WAP) bir sinyal gücü (RSSI) yayar (-104 ile 0 dBm arası)
2. Telefonun aldığı 520 WAP sinyali bir "parmak izi" oluşturur
3. Her konum (bina + kat) kendine özgü bir sinyal parmak izi bırakır
4. Model bu parmak izlerini öğrenerek yeni bir ölçümden konum tahmin eder

LightGBM nedir?
- Gradient Boosting = birçok KÜÇÜK karar ağacını SIRALI olarak eğitmek
- Her yeni ağaç, önceki ağaçların HATALARINI düzeltmeye odaklanır
- Random Forest'tan farkı: ağaçlar paralel değil, sıralı ve birbirine bağlı
- "Light" = histogram tabanlı bölme ile çok daha hızlı
- Sparse (seyrek) veri ile mükemmel çalışır (bizim verimiz %96.5 boş!)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. VERİYİ YÜKLE
# ============================================================
print("=" * 60)
print("1. VERİ YÜKLEME")
print("=" * 60)

train = pd.read_csv("archive-10/TrainingData.csv")
test = pd.read_csv("archive-10/ValidationData.csv")

print(f"Eğitim seti  : {train.shape[0]} satır, {train.shape[1]} sütun")
print(f"Test seti     : {test.shape[0]} satır, {test.shape[1]} sütun")

# WAP sütunları = özellikler (features), geri kalanı = meta veri + hedef
wap_cols = [col for col in train.columns if col.startswith("WAP")]
print(f"WAP sayısı    : {len(wap_cols)}")
print(f"Binalar       : {sorted(train['BUILDINGID'].unique())}")
print(f"Katlar        : {sorted(train['FLOOR'].unique())}")

# ============================================================
# 2. VERİ ÖN İŞLEME
# ============================================================
print("\n" + "=" * 60)
print("2. VERİ ÖN İŞLEME")
print("=" * 60)

# 100 değeri = "sinyal algılanmadı" demek
# Bunu -105'e çeviriyoruz (gerçek minimum -104'ten düşük bir değer)
# Böylece model "sinyal yok" durumunu en zayıf sinyal olarak algılar
X_train = train[wap_cols].replace(100, -105).values
X_test = test[wap_cols].replace(100, -105).values

print(f"Sinyal yok (100) -> -105'e dönüştürüldü")
print(f"Sinyal aralığı: {X_train[X_train != -105].min():.0f} ile {X_train[X_train != -105].max():.0f} dBm")

# ============================================================
# 3. HEDEF DEĞİŞKEN: BİNA + KAT BİRLEŞİK ETİKET
# ============================================================
print("\n" + "=" * 60)
print("3. HEDEF DEĞİŞKEN OLUŞTURMA")
print("=" * 60)

# Bina ve katı tek bir etiket olarak birleştiriyoruz
# Örnek: Bina 1, Kat 3 -> "1_3"
# Bu sayede tek model hem binayı hem katı aynı anda tahmin eder
train["LABEL"] = train["BUILDINGID"].astype(str) + "_" + train["FLOOR"].astype(str)
test["LABEL"] = test["BUILDINGID"].astype(str) + "_" + test["FLOOR"].astype(str)

le = LabelEncoder()
y_train = le.fit_transform(train["LABEL"])
y_test = le.transform(test["LABEL"])

print(f"Toplam sınıf sayısı: {len(le.classes_)}")
print(f"Sınıflar (Bina_Kat): {list(le.classes_)}")

# Her sınıftaki örnek sayısı
print("\nSınıf dağılımı:")
for cls in sorted(train["LABEL"].unique()):
    bina, kat = cls.split("_")
    count = (train["LABEL"] == cls).sum()
    print(f"  Bina {bina}, Kat {kat}: {count} örnek")

# ============================================================
# 4. LightGBM MODELİ
# ============================================================
print("\n" + "=" * 60)
print("4. MODEL EĞİTİMİ (LightGBM)")
print("=" * 60)

"""
LightGBM Parametreleri (basit açıklama):

- num_leaves=63       : Her ağaçtaki yaprak sayısı (karmaşıklık kontrolü)
                         Fazlaysa -> daha detaylı öğrenir ama ezberleyebilir
- max_depth=8         : Ağaç derinliği (kaç kez bölme yapılabilir)
- learning_rate=0.05  : Öğrenme hızı (küçük = yavaş ama stabil öğrenme)
- n_estimators=500    : Kaç tane ağaç kullanılacak
- min_child_samples=10: Bir yaprakta minimum örnek sayısı (ezberi önler)
- subsample=0.8       : Her ağaç için verinin %80'ini rastgele seç
- colsample_bytree=0.8: Her ağaç için sütunların %80'ini rastgele seç
- reg_alpha=0.1       : L1 regularizasyon (gereksiz özellikleri sıfırlar)
- reg_lambda=0.1      : L2 regularizasyon (büyük ağırlıkları cezalandırır)
"""

model = lgb.LGBMClassifier(
    num_leaves=63,
    max_depth=8,
    learning_rate=0.05,
    n_estimators=500,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,           # Eğitim loglarını gizle
    n_jobs=-1,            # Tüm CPU çekirdeklerini kullan
    class_weight="balanced"  # Az örnekli sınıflara daha fazla ağırlık ver
)

print("Model eğitiliyor...")
model.fit(X_train, y_train)
print("Model eğitimi tamamlandı!")

# ============================================================
# 5. TAHMİN VE DEĞERLENDİRME
# ============================================================
print("\n" + "=" * 60)
print("5. SONUÇLAR")
print("=" * 60)

y_pred = model.predict(X_test)

# Genel doğruluk
accuracy = accuracy_score(y_test, y_pred)
print(f"\n{'='*40}")
print(f"  GENEL DOĞRULUK: %{accuracy * 100:.1f}")
print(f"{'='*40}")

# Tahminleri geri çöz -> Bina ve Kat olarak ayır
pred_labels = le.inverse_transform(y_pred)
true_labels = le.inverse_transform(y_test)

pred_building = [int(l.split("_")[0]) for l in pred_labels]
pred_floor = [int(l.split("_")[1]) for l in pred_labels]
true_building = [int(l.split("_")[0]) for l in true_labels]
true_floor = [int(l.split("_")[1]) for l in true_labels]

# Bina doğruluğu
building_acc = accuracy_score(true_building, pred_building)
print(f"  BİNA DOĞRULUĞU: %{building_acc * 100:.1f}")

# Kat doğruluğu
floor_acc = accuracy_score(true_floor, pred_floor)
print(f"  KAT DOĞRULUĞU : %{floor_acc * 100:.1f}")

# Detaylı rapor
print("\n--- Detaylı Sınıflandırma Raporu ---")
target_names = [f"Bina {c.split('_')[0]} Kat {c.split('_')[1]}" for c in le.classes_]
print(classification_report(y_test, y_pred, target_names=target_names))

# ============================================================
# 6. GÖRSELLEŞTİRME
# ============================================================
print("\n" + "=" * 60)
print("6. GÖRSELLEŞTİRME")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle("WiFi Fingerprinting - LightGBM Sonuçları", fontsize=16, fontweight="bold")

# --- 6a. Bina Confusion Matrix ---
cm_building = confusion_matrix(true_building, pred_building)
sns.heatmap(cm_building, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0],
            xticklabels=["Bina 0", "Bina 1", "Bina 2"],
            yticklabels=["Bina 0", "Bina 1", "Bina 2"])
axes[0, 0].set_title(f"Bina Tespiti (Doğruluk: %{building_acc*100:.1f})")
axes[0, 0].set_ylabel("Gerçek")
axes[0, 0].set_xlabel("Tahmin")

# --- 6b. Kat Confusion Matrix ---
cm_floor = confusion_matrix(true_floor, pred_floor)
sns.heatmap(cm_floor, annot=True, fmt="d", cmap="Greens", ax=axes[0, 1],
            xticklabels=[f"Kat {i}" for i in range(5)],
            yticklabels=[f"Kat {i}" for i in range(5)])
axes[0, 1].set_title(f"Kat Tespiti (Doğruluk: %{floor_acc*100:.1f})")
axes[0, 1].set_ylabel("Gerçek")
axes[0, 1].set_xlabel("Tahmin")

# --- 6c. En Önemli 20 WAP ---
importance = model.feature_importances_
top_20_idx = np.argsort(importance)[-20:]
top_20_names = [wap_cols[i] for i in top_20_idx]
top_20_values = importance[top_20_idx]

axes[1, 0].barh(range(20), top_20_values, color="coral")
axes[1, 0].set_yticks(range(20))
axes[1, 0].set_yticklabels(top_20_names, fontsize=9)
axes[1, 0].set_title("En Önemli 20 WiFi Access Point")
axes[1, 0].set_xlabel("Önem Skoru (Karar ağaçlarında kullanım sayısı)")

# --- 6d. Bina bazında kat dağılımı ---
for bina in [0, 1, 2]:
    mask = np.array(true_building) == bina
    if mask.sum() > 0:
        correct = np.array(true_floor)[mask] == np.array(pred_floor)[mask]
        floors = sorted(set(np.array(true_floor)[mask]))
        floor_accs = []
        for f in floors:
            f_mask = mask & (np.array(true_floor) == f)
            if f_mask.sum() > 0:
                f_acc = (np.array(pred_floor)[f_mask] == f).mean() * 100
                floor_accs.append(f_acc)
            else:
                floor_accs.append(0)
        axes[1, 1].bar([f + bina * 0.25 - 0.25 for f in floors], floor_accs,
                       width=0.25, label=f"Bina {bina}", alpha=0.8)

axes[1, 1].set_title("Bina Bazında Kat Doğruluk Oranları")
axes[1, 1].set_xlabel("Kat")
axes[1, 1].set_ylabel("Doğruluk (%)")
axes[1, 1].set_xticks(range(5))
axes[1, 1].set_xticklabels([f"Kat {i}" for i in range(5)])
axes[1, 1].legend()
axes[1, 1].set_ylim(0, 105)

plt.tight_layout()
plt.savefig("sonuclar.png", dpi=150, bbox_inches="tight")
print("Grafik 'sonuclar.png' olarak kaydedildi!")
plt.show()

# ============================================================
# 7. ÖZET
# ============================================================
print("\n" + "=" * 60)
print("7. MODEL NASIL ÇALIŞIYOR? (ÖZET)")
print("=" * 60)
print("""
┌─────────────────────────────────────────────────────────┐
│                  LightGBM ÇALIŞMA MANTIGI               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Telefonun ölçtüğü 520 WAP sinyali                     │
│         │                                               │
│         ▼                                               │
│  ┌─── Ağaç 1 ───┐                                      │
│  │ "WAP171 < -60 │ → İlk kaba tahmin                   │
│  │  ise Bina 1"  │                                      │
│  └───────────────┘                                      │
│         │ hata                                          │
│         ▼                                               │
│  ┌─── Ağaç 2 ───┐                                      │
│  │ "WAP103 < -80 │ → Ağaç 1'in hatasını düzelt         │
│  │  ise Kat 2"   │                                      │
│  └───────────────┘                                      │
│         │ kalan hata                                    │
│         ▼                                               │
│       ...                                               │
│  ┌─── Ağaç 500 ──┐                                     │
│  │ Son düzeltme   │ → İnce ayar                         │
│  └────────────────┘                                     │
│         │                                               │
│         ▼                                               │
│  Tüm ağaçların tahminlerini TOPLA → Final tahmin        │
│                                                         │
│  Random Forest: ağaçlar BAĞIMSIZ, çoğunluk oyu          │
│  LightGBM:      ağaçlar SIRALI, her biri hatayı düzeltir│
│                                                         │
└─────────────────────────────────────────────────────────┘
""")

# ============================================================
# 8. BİNA SİMÜLASYONU VE TEST
# ============================================================
print("\n" + "=" * 60)
print("8. BİNA SİMÜLASYONU")
print("=" * 60)

"""
SİMÜLASYON MANTIGI:
  - 3 binayı ve katlarını sanal olarak oluşturuyoruz
  - Her kata rastgele "kişiler" yerleştiriyoruz
  - Her kişi için gerçek datasetten o bina+kat'a ait WiFi parmak izi alıyoruz
  - Sinyale gürültü ekliyoruz (gerçek hayattaki dalgalanma)
  - Model bu gürültülü sinyallerden bina ve kat tahmin ediyor
  - Doğru/yanlış tahminleri bina kesit görünümünde gösteriyoruz
"""

np.random.seed(42)

# Her bina için kat sayıları (datasetteki gibi)
bina_kat_sayilari = {0: 4, 1: 4, 2: 5}  # Bina 0: Kat 0-3, Bina 2: Kat 0-4
bina_isimleri = {0: "Mühendislik\nFakültesi", 1: "Fen\nFakültesi", 2: "Kütüphane"}
bina_renkleri = {0: "#4A90D9", 1: "#E67E22", 2: "#2ECC71"}

# Her bina+kat için gerçek veriden örnekler seçip gürültü ekle
kisi_sayisi_per_kat = 3  # Her kata 3 kişi koy

simulasyon_verileri = []  # (bina, kat, x_pozisyon, wifi_sinyal, tahmin_bina, tahmin_kat)

for bina_id, kat_sayisi in bina_kat_sayilari.items():
    for kat in range(kat_sayisi):
        # Bu bina+kat'a ait gerçek verilerden rastgele seç
        mask = (train["BUILDINGID"] == bina_id) & (train["FLOOR"] == kat)
        gercek_ornekler = train[mask]

        if len(gercek_ornekler) < kisi_sayisi_per_kat:
            continue

        # Rastgele kişiler seç
        secilen = gercek_ornekler.sample(n=kisi_sayisi_per_kat, random_state=42 + bina_id * 10 + kat)

        for i, (idx, row) in enumerate(secilen.iterrows()):
            # WiFi sinyalini al ve gürültü ekle (gerçek hayat simülasyonu)
            sinyal = row[wap_cols].values.astype(float).copy()

            # Gürültü: algılanan sinyallere ±5 dBm rastgele sapma ekle
            algilanan = sinyal != 100
            gurultu = np.random.normal(0, 5, size=sinyal.shape)
            sinyal[algilanan] = np.clip(sinyal[algilanan] + gurultu[algilanan], -104, 0)
            sinyal[~algilanan] = 100  # Algılanmayanlar hala 100

            # Modele ver (100 -> -105 dönüşümü)
            sinyal_model = np.where(sinyal == 100, -105, sinyal).reshape(1, -1)
            tahmin = model.predict(sinyal_model)[0]
            tahmin_label = le.inverse_transform([tahmin])[0]
            tahmin_bina = int(tahmin_label.split("_")[0])
            tahmin_kat = int(tahmin_label.split("_")[1])

            # Kişinin kattaki x pozisyonu (görselleştirme için)
            x_poz = (i + 1) / (kisi_sayisi_per_kat + 1)

            simulasyon_verileri.append({
                "gercek_bina": bina_id,
                "gercek_kat": kat,
                "tahmin_bina": tahmin_bina,
                "tahmin_kat": tahmin_kat,
                "x_poz": x_poz,
                "dogru": (bina_id == tahmin_bina) and (kat == tahmin_kat),
                "aktif_wap": int(algilanan.sum()),
                "max_sinyal": float(sinyal[algilanan].max()) if algilanan.any() else -105,
            })

sim_df = pd.DataFrame(simulasyon_verileri)
sim_dogru = sim_df["dogru"].sum()
sim_toplam = len(sim_df)
print(f"  Toplam simüle edilen kişi : {sim_toplam}")
print(f"  Doğru tahmin              : {sim_dogru}/{sim_toplam} (%{sim_dogru/sim_toplam*100:.0f})")
print(f"  Yanlış tahmin             : {sim_toplam - sim_dogru}/{sim_toplam}")

# ============================================================
# 9. BİNA KESİT GÖRSELLEŞTİRME
# ============================================================
print("\n" + "=" * 60)
print("9. BİNA SİMÜLASYON GÖRSELLEŞTİRMESİ")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(22, 10))
fig.suptitle("Bina Simülasyonu - WiFi ile Kat Tespiti Testi",
             fontsize=18, fontweight="bold", y=1.02)

kat_yukseklik = 1.0  # Her kat görsel yüksekliği
bina_genislik = 3.0  # Bina görsel genişliği

for bina_id, ax in enumerate(axes):
    kat_sayisi = bina_kat_sayilari[bina_id]

    # --- Bina çerçevesini çiz ---
    # Zemin
    ax.fill_between([-0.3, bina_genislik + 0.3], [-0.3, -0.3], [0, 0],
                    color="#8B7355", alpha=0.5)

    for kat in range(kat_sayisi):
        y_alt = kat * kat_yukseklik
        y_ust = (kat + 1) * kat_yukseklik

        # Kat zemini/tavanı
        ax.fill_between([0, bina_genislik], [y_alt, y_alt], [y_ust, y_ust],
                        color=bina_renkleri[bina_id], alpha=0.08)

        # Kat çizgileri (döşeme)
        ax.plot([0, bina_genislik], [y_alt, y_alt], color="gray", linewidth=2)

        # Pencereler
        for px in [0.4, 1.2, 2.0, 2.6]:
            pencere_y = y_alt + 0.25
            ax.fill([px, px + 0.3, px + 0.3, px],
                    [pencere_y, pencere_y, pencere_y + 0.45, pencere_y + 0.45],
                    color="lightskyblue", edgecolor="gray", linewidth=0.8, alpha=0.5)

        # Kat etiketi
        ax.text(-0.25, y_alt + kat_yukseklik / 2, f"Kat {kat}",
                fontsize=10, fontweight="bold", va="center", ha="right",
                color="#333333")

    # Çatı
    y_cati = kat_sayisi * kat_yukseklik
    ax.plot([0, bina_genislik], [y_cati, y_cati], color="gray", linewidth=2)
    # Çatı üçgeni
    ax.fill([0, bina_genislik / 2, bina_genislik],
            [y_cati, y_cati + 0.5, y_cati],
            color=bina_renkleri[bina_id], alpha=0.3, edgecolor="gray", linewidth=2)

    # Duvarlar
    ax.plot([0, 0], [0, y_cati], color="gray", linewidth=2.5)
    ax.plot([bina_genislik, bina_genislik], [0, y_cati], color="gray", linewidth=2.5)

    # WiFi router ikonları (her kata 1 tane)
    for kat in range(kat_sayisi):
        router_y = kat * kat_yukseklik + 0.8
        ax.plot(bina_genislik - 0.15, router_y, marker="^", markersize=8,
                color="red", alpha=0.6, zorder=5)
        # Sinyal dalgaları
        for r in [0.12, 0.22, 0.32]:
            circle = plt.Circle((bina_genislik - 0.15, router_y), r,
                                fill=False, color="red", alpha=0.2, linewidth=0.8)
            ax.add_patch(circle)

    # --- Kişileri yerleştir ---
    bina_sim = sim_df[sim_df["gercek_bina"] == bina_id]

    dogru_sayisi = 0
    yanlis_sayisi = 0

    for _, kisi in bina_sim.iterrows():
        kat = kisi["gercek_kat"]
        x = kisi["x_poz"] * bina_genislik
        y = kat * kat_yukseklik + 0.05  # Zeminin hemen üstü

        if kisi["dogru"]:
            # DOĞRU tahmin -> yeşil kişi
            renk = "#27AE60"
            ikon = "o"  # dolu daire = kişi kafası
            dogru_sayisi += 1
            etiket = f"Kat {kisi['tahmin_kat']}"
        else:
            # YANLIŞ tahmin -> kırmızı kişi
            renk = "#E74C3C"
            ikon = "X"
            yanlis_sayisi += 1
            if kisi["tahmin_bina"] != bina_id:
                etiket = f"B{kisi['tahmin_bina']}K{kisi['tahmin_kat']}"
            else:
                etiket = f"Kat {kisi['tahmin_kat']}"

        # Kişi gövdesi (çubuk adam)
        ax.plot(x, y + 0.22, marker=ikon, markersize=14,
                color=renk, zorder=10, markeredgecolor="white", markeredgewidth=1.5)
        ax.plot([x, x], [y + 0.02, y + 0.16], color=renk, linewidth=2.5, zorder=9)
        # Kollar
        ax.plot([x - 0.12, x + 0.12], [y + 0.12, y + 0.12],
                color=renk, linewidth=2, zorder=9)
        # Bacaklar
        ax.plot([x, x - 0.08], [y + 0.02, y - 0.02],
                color=renk, linewidth=2, zorder=9)
        ax.plot([x, x + 0.08], [y + 0.02, y - 0.02],
                color=renk, linewidth=2, zorder=9)

        # Tahmin baloncuğu
        ax.annotate(etiket, xy=(x, y + 0.32), fontsize=7,
                    ha="center", va="bottom", fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=renk, alpha=0.85))

    # Başlık
    ax.set_title(f"{bina_isimleri[bina_id]}\n({dogru_sayisi} dogru / "
                 f"{yanlis_sayisi} yanlis)",
                 fontsize=13, fontweight="bold", color=bina_renkleri[bina_id], pad=15)

    # Eksen ayarları
    ax.set_xlim(-0.6, bina_genislik + 0.6)
    ax.set_ylim(-0.5, kat_sayisi * kat_yukseklik + 1.0)
    ax.set_aspect("equal")
    ax.axis("off")

# Lejant
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#27AE60",
           markersize=14, label="Dogru Tahmin"),
    Line2D([0], [0], marker="X", color="w", markerfacecolor="#E74C3C",
           markersize=14, label="Yanlis Tahmin"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="red",
           markersize=10, label="WiFi Router", alpha=0.6),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=3,
           fontsize=12, frameon=True, fancybox=True, shadow=True,
           bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.savefig("simulasyon.png", dpi=150, bbox_inches="tight")
print("Bina simülasyonu 'simulasyon.png' olarak kaydedildi!")
plt.show()

# ============================================================
# 10. SİMÜLASYON DETAY TABLOSU
# ============================================================
print("\n" + "=" * 60)
print("10. SİMÜLASYON DETAYLARI")
print("=" * 60)

print(f"\n{'Kisi':>4} {'Gercek':>12} {'Tahmin':>12} {'Sonuc':>8} {'Aktif WAP':>10} {'Max Sinyal':>11}")
print("-" * 60)
for i, row in sim_df.iterrows():
    gercek = f"B{row['gercek_bina']}K{row['gercek_kat']}"
    tahmin = f"B{row['tahmin_bina']}K{row['tahmin_kat']}"
    sonuc = "DOGRU" if row["dogru"] else "YANLIS"
    renk_kod = "" if row["dogru"] else " <--"
    print(f"  {i+1:>2}   {gercek:>10}   {tahmin:>10}   {sonuc:>7}   {row['aktif_wap']:>8}   {row['max_sinyal']:>8.0f} dBm{renk_kod}")

print(f"\n  Simülasyon Özeti: {sim_dogru}/{sim_toplam} doğru (%{sim_dogru/sim_toplam*100:.0f})")
