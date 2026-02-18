"""
WiFi Fingerprinting - İnteraktif Bina & Kat Tespiti Demo
=========================================================
Streamlit ile çalışan interaktif web uygulaması.
Çalıştır: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# SAYFA AYARLARI
# ============================================================
st.set_page_config(
    page_title="WiFi ile Konum Tespiti",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card h2 {
        margin: 0;
        font-size: 2.2rem;
    }
    .metric-card p {
        margin: 0.3rem 0 0 0;
        opacity: 0.9;
        font-size: 0.9rem;
    }
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .warning-card {
        background: linear-gradient(135deg, #F2994A 0%, #F2C94C 100%);
    }
    .info-card {
        background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# VERİ VE MODEL YÜKLEME (cache ile sadece 1 kez çalışır)
# ============================================================
@st.cache_resource
def load_model():
    """Veriyi yükle ve modeli eğit (sadece ilk açılışta çalışır)."""
    train = pd.read_csv("archive-10/TrainingData.csv")
    test = pd.read_csv("archive-10/ValidationData.csv")

    wap_cols = [col for col in train.columns if col.startswith("WAP")]

    X_train = train[wap_cols].replace(100, -105).values
    X_test = test[wap_cols].replace(100, -105).values

    train["LABEL"] = train["BUILDINGID"].astype(str) + "_" + train["FLOOR"].astype(str)
    test["LABEL"] = test["BUILDINGID"].astype(str) + "_" + test["FLOOR"].astype(str)

    le = LabelEncoder()
    y_train = le.fit_transform(train["LABEL"])
    y_test = le.transform(test["LABEL"])

    model = lgb.LGBMClassifier(
        num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=500,
        min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1,
        n_jobs=-1, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return model, le, train, test, wap_cols, X_test, y_test, y_pred


model, le, train, test, wap_cols, X_test, y_test, y_pred = load_model()

# Sabitler
BINA_ISIMLERI = {0: "Mühendislik Fakültesi", 1: "Fen Fakültesi", 2: "Kütüphane"}
BINA_KATLARI = {0: [0, 1, 2, 3], 1: [0, 1, 2, 3], 2: [0, 1, 2, 3, 4]}
BINA_RENKLERI = {0: "#4A90D9", 1: "#E67E22", 2: "#2ECC71"}

# ============================================================
# BAŞLIK
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>📡 WiFi Fingerprinting ile Bina & Kat Tespiti</h1>
    <p style="font-size: 1.1rem; color: #666;">
        520 WiFi Access Point sinyalinden konum tahmin eden LightGBM modeli
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🏢 Canlı Simülasyon",
    "📊 Model Performansı",
    "🔬 Sinyal Analizi",
    "🧠 Nasıl Çalışıyor?"
])

# ============================================================
# TAB 1: CANLI SİMÜLASYON
# ============================================================
with tab1:
    st.markdown("### 🎮 Bina İçinde Yürüyüş Simülasyonu")
    st.markdown("Bir bina ve kat seçin, model WiFi sinyallerinden konumunuzu tahmin etsin!")

    col_control, col_viz = st.columns([1, 2.5])

    with col_control:
        st.markdown("#### ⚙️ Kontroller")

        # Bina seçimi
        secili_bina = st.selectbox(
            "🏢 Bina Seçin",
            options=[0, 1, 2],
            format_func=lambda x: f"Bina {x} - {BINA_ISIMLERI[x]}"
        )

        # Kat seçimi
        secili_kat = st.selectbox(
            "🏗️ Gerçek Katınız",
            options=BINA_KATLARI[secili_bina],
            format_func=lambda x: f"Kat {x}"
        )

        # Gürültü seviyesi
        gurultu = st.slider(
            "📶 Sinyal Gürültüsü (dBm)",
            min_value=0, max_value=20, value=5,
            help="Gerçek hayattaki sinyal dalgalanması. 0 = ideal, 20 = çok gürültülü"
        )

        # Kişi sayısı
        kisi_sayisi = st.slider(
            "👥 Simüle Edilecek Kişi",
            min_value=1, max_value=10, value=5
        )

        # Test butonu
        test_et = st.button("🚀 Simülasyonu Başlat", use_container_width=True, type="primary")

    with col_viz:
        if test_et:
            # Gerçek veriden örnekler al
            mask = (train["BUILDINGID"] == secili_bina) & (train["FLOOR"] == secili_kat)
            ornekler = train[mask]

            if len(ornekler) < kisi_sayisi:
                st.error("Bu bina+kat için yeterli veri yok!")
            else:
                np.random.seed(None)  # Her tıklamada farklı sonuç
                secilen = ornekler.sample(n=kisi_sayisi)

                sonuclar = []
                for idx, row in secilen.iterrows():
                    sinyal = row[wap_cols].values.astype(float).copy()
                    algilanan = sinyal != 100

                    # Gürültü ekle
                    if gurultu > 0:
                        noise = np.random.normal(0, gurultu, size=sinyal.shape)
                        sinyal[algilanan] = np.clip(
                            sinyal[algilanan] + noise[algilanan], -104, 0
                        )

                    sinyal_model = np.where(sinyal == 100, -105, sinyal).reshape(1, -1)
                    tahmin = model.predict(sinyal_model)[0]
                    proba = model.predict_proba(sinyal_model)[0]
                    tahmin_label = le.inverse_transform([tahmin])[0]
                    t_bina, t_kat = int(tahmin_label.split("_")[0]), int(tahmin_label.split("_")[1])

                    sonuclar.append({
                        "tahmin_bina": t_bina,
                        "tahmin_kat": t_kat,
                        "dogru": (t_bina == secili_bina) and (t_kat == secili_kat),
                        "guven": float(proba.max()) * 100,
                        "aktif_wap": int(algilanan.sum()),
                    })

                # ---- BİNA GÖRSELLEŞTİRME (Plotly) ----
                fig = go.Figure()

                kat_sayisi = len(BINA_KATLARI[secili_bina])
                bina_w, kat_h = 6, 1.2

                # Kat zeminleri
                for k in range(kat_sayisi):
                    y0 = k * kat_h
                    renk = BINA_RENKLERI[secili_bina]

                    # Seçili kat vurgusu
                    if k == secili_kat:
                        opacity = 0.25
                        line_w = 3
                    else:
                        opacity = 0.06
                        line_w = 1

                    fig.add_shape(type="rect",
                        x0=0, y0=y0, x1=bina_w, y1=y0 + kat_h,
                        fillcolor=renk, opacity=opacity,
                        line=dict(color=renk, width=line_w))

                    # Kat etiketi
                    fig.add_annotation(x=-0.4, y=y0 + kat_h / 2,
                        text=f"<b>Kat {k}</b>", showarrow=False,
                        font=dict(size=13, color="#333"))

                # Çatı
                y_top = kat_sayisi * kat_h
                fig.add_shape(type="line", x0=0, y0=y_top, x1=bina_w, y1=y_top,
                    line=dict(color="gray", width=2))

                # WiFi routerlar
                for k in range(kat_sayisi):
                    fig.add_trace(go.Scatter(
                        x=[bina_w - 0.3], y=[k * kat_h + kat_h * 0.75],
                        mode="markers+text", text=["📡"], textposition="middle center",
                        textfont=dict(size=16),
                        marker=dict(size=1, color="rgba(0,0,0,0)"),
                        showlegend=False, hoverinfo="skip"
                    ))

                # Kişileri yerleştir
                dogru_x, dogru_y, dogru_text = [], [], []
                yanlis_x, yanlis_y, yanlis_text = [], [], []

                for i, s in enumerate(sonuclar):
                    x_pos = (i + 1) * bina_w / (kisi_sayisi + 1)
                    y_pos = secili_kat * kat_h + kat_h * 0.45

                    hover = (f"Kişi {i+1}<br>"
                             f"Tahmin: Bina {s['tahmin_bina']} Kat {s['tahmin_kat']}<br>"
                             f"Güven: %{s['guven']:.0f}<br>"
                             f"Aktif WAP: {s['aktif_wap']}")

                    if s["dogru"]:
                        dogru_x.append(x_pos)
                        dogru_y.append(y_pos)
                        dogru_text.append(hover)
                    else:
                        yanlis_x.append(x_pos)
                        yanlis_y.append(y_pos)
                        yanlis_text.append(hover)

                if dogru_x:
                    fig.add_trace(go.Scatter(
                        x=dogru_x, y=dogru_y, mode="markers",
                        marker=dict(size=28, color="#27AE60", symbol="circle",
                                    line=dict(width=2, color="white")),
                        name=f"✅ Doğru ({len(dogru_x)})",
                        text=dogru_text, hoverinfo="text"
                    ))

                if yanlis_x:
                    fig.add_trace(go.Scatter(
                        x=yanlis_x, y=yanlis_y, mode="markers",
                        marker=dict(size=28, color="#E74C3C", symbol="x",
                                    line=dict(width=3, color="white")),
                        name=f"❌ Yanlış ({len(yanlis_x)})",
                        text=yanlis_text, hoverinfo="text"
                    ))

                fig.update_layout(
                    title=dict(
                        text=f"<b>{BINA_ISIMLERI[secili_bina]}</b> - Kat {secili_kat} Simülasyonu",
                        font=dict(size=18)
                    ),
                    xaxis=dict(visible=False, range=[-1, bina_w + 1]),
                    yaxis=dict(visible=False, range=[-0.5, kat_sayisi * kat_h + 0.8],
                               scaleanchor="x"),
                    height=450,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.15, x=0.5, xanchor="center",
                                font=dict(size=14)),
                    margin=dict(l=40, r=20, t=60, b=60),
                    plot_bgcolor="white"
                )

                st.plotly_chart(fig, use_container_width=True)

                # Sonuç metrikleri
                dogru_sayi = sum(1 for s in sonuclar if s["dogru"])
                ort_guven = np.mean([s["guven"] for s in sonuclar])

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f"""
                    <div class="metric-card success-card">
                        <h2>{dogru_sayi}/{kisi_sayisi}</h2>
                        <p>Doğru Tahmin</p>
                    </div>""", unsafe_allow_html=True)
                with m2:
                    acc = dogru_sayi / kisi_sayisi * 100
                    card_class = "success-card" if acc >= 80 else "warning-card"
                    st.markdown(f"""
                    <div class="metric-card {card_class}">
                        <h2>%{acc:.0f}</h2>
                        <p>Doğruluk Oranı</p>
                    </div>""", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""
                    <div class="metric-card info-card">
                        <h2>%{ort_guven:.0f}</h2>
                        <p>Ortalama Güven</p>
                    </div>""", unsafe_allow_html=True)

                # Detay tablosu
                st.markdown("#### 📋 Detaylı Sonuçlar")
                tablo = pd.DataFrame(sonuclar)
                tablo.index = [f"Kişi {i+1}" for i in range(len(tablo))]
                tablo.columns = ["Tahmin Bina", "Tahmin Kat", "Doğru?", "Güven %", "Aktif WAP"]
                tablo["Doğru?"] = tablo["Doğru?"].map({True: "✅", False: "❌"})
                tablo["Güven %"] = tablo["Güven %"].apply(lambda x: f"%{x:.1f}")
                st.dataframe(tablo, use_container_width=True)

        else:
            # Başlangıç görseli
            st.info("👆 Soldaki kontrolleri ayarlayıp **Simülasyonu Başlat** butonuna tıklayın!")

            # Kampüs haritası
            fig = go.Figure()

            positions = {0: (1, 1), 1: (4, 1), 2: (2.5, 3.5)}
            sizes = {0: 4, 1: 4, 2: 5}

            for bina_id, (x, y) in positions.items():
                fig.add_trace(go.Scatter(
                    x=[x], y=[y], mode="markers+text",
                    marker=dict(size=60 + sizes[bina_id] * 8,
                                color=BINA_RENKLERI[bina_id], opacity=0.3,
                                line=dict(width=3, color=BINA_RENKLERI[bina_id])),
                    text=[f"🏢<br><b>{BINA_ISIMLERI[bina_id]}</b><br>{sizes[bina_id]} kat"],
                    textposition="middle center",
                    textfont=dict(size=11),
                    name=BINA_ISIMLERI[bina_id],
                    showlegend=False, hoverinfo="text",
                    hovertext=f"{BINA_ISIMLERI[bina_id]}\n{sizes[bina_id]} kat"
                ))

            fig.update_layout(
                title="<b>🏫 Kampüs Haritası</b>",
                xaxis=dict(visible=False, range=[-0.5, 5.5]),
                yaxis=dict(visible=False, range=[-0.5, 5.5], scaleanchor="x"),
                height=400, plot_bgcolor="#F8F9FA",
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 2: MODEL PERFORMANSI
# ============================================================
with tab2:
    st.markdown("### 📊 Model Performans Analizi")

    pred_labels = le.inverse_transform(y_pred)
    true_labels = le.inverse_transform(y_test)
    p_building = np.array([int(l.split("_")[0]) for l in pred_labels])
    p_floor = np.array([int(l.split("_")[1]) for l in pred_labels])
    t_building = np.array([int(l.split("_")[0]) for l in true_labels])
    t_floor = np.array([int(l.split("_")[1]) for l in true_labels])

    building_acc = accuracy_score(t_building, p_building) * 100
    floor_acc = accuracy_score(t_floor, p_floor) * 100
    overall_acc = accuracy_score(y_test, y_pred) * 100

    # Metrikler
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <h2>%{building_acc:.1f}</h2>
            <p>🏢 Bina Doğruluğu</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card success-card">
            <h2>%{floor_acc:.1f}</h2>
            <p>🏗️ Kat Doğruluğu</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card info-card">
            <h2>%{overall_acc:.1f}</h2>
            <p>📍 Genel Doğruluk</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    col_cm1, col_cm2 = st.columns(2)

    # Bina Confusion Matrix
    with col_cm1:
        cm_b = confusion_matrix(t_building, p_building)
        fig_b = go.Figure(data=go.Heatmap(
            z=cm_b, x=["Bina 0", "Bina 1", "Bina 2"],
            y=["Bina 0", "Bina 1", "Bina 2"],
            colorscale="Blues", texttemplate="%{z}",
            textfont=dict(size=16),
            hovertemplate="Gerçek: %{y}<br>Tahmin: %{x}<br>Sayı: %{z}<extra></extra>"
        ))
        fig_b.update_layout(title="<b>Bina Confusion Matrix</b>",
                            xaxis_title="Tahmin", yaxis_title="Gerçek",
                            height=380, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_b, use_container_width=True)

    # Kat Confusion Matrix
    with col_cm2:
        cm_f = confusion_matrix(t_floor, p_floor)
        kat_labels = [f"Kat {i}" for i in range(5)]
        fig_f = go.Figure(data=go.Heatmap(
            z=cm_f, x=kat_labels, y=kat_labels,
            colorscale="Greens", texttemplate="%{z}",
            textfont=dict(size=16),
            hovertemplate="Gerçek: %{y}<br>Tahmin: %{x}<br>Sayı: %{z}<extra></extra>"
        ))
        fig_f.update_layout(title="<b>Kat Confusion Matrix</b>",
                            xaxis_title="Tahmin", yaxis_title="Gerçek",
                            height=380, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_f, use_container_width=True)

    # Bina bazında kat doğrulukları
    st.markdown("#### 🏢 Bina Bazında Kat Doğrulukları")
    fig_bar = go.Figure()
    for bina in [0, 1, 2]:
        b_mask = t_building == bina
        floors = sorted(np.unique(t_floor[b_mask]))
        accs = []
        for f in floors:
            f_mask = b_mask & (t_floor == f)
            accs.append((p_floor[f_mask] == f).mean() * 100 if f_mask.sum() > 0 else 0)
        fig_bar.add_trace(go.Bar(
            x=[f"Kat {f}" for f in floors], y=accs,
            name=BINA_ISIMLERI[bina],
            marker_color=BINA_RENKLERI[bina],
            text=[f"%{a:.0f}" for a in accs], textposition="outside"
        ))
    fig_bar.update_layout(barmode="group", yaxis_range=[0, 110],
                          yaxis_title="Doğruluk (%)", height=400,
                          legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"))
    st.plotly_chart(fig_bar, use_container_width=True)


# ============================================================
# TAB 3: SİNYAL ANALİZİ
# ============================================================
with tab3:
    st.markdown("### 🔬 WiFi Sinyal Analizi")

    col_s1, col_s2 = st.columns([1, 1])

    with col_s1:
        # En önemli WAP'lar
        st.markdown("#### 📡 En Önemli 20 WiFi Access Point")
        importance = model.feature_importances_
        top_idx = np.argsort(importance)[-20:][::-1]
        top_names = [wap_cols[i] for i in top_idx]
        top_vals = importance[top_idx]

        fig_imp = go.Figure(go.Bar(
            x=top_vals[::-1], y=top_names[::-1], orientation="h",
            marker=dict(color=top_vals[::-1], colorscale="YlOrRd"),
            hovertemplate="%{y}: %{x} kullanım<extra></extra>"
        ))
        fig_imp.update_layout(height=500, xaxis_title="Önem Skoru",
                              margin=dict(l=80))
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_s2:
        # İnteraktif: Bir WAP'ın binalardaki sinyal dağılımı
        st.markdown("#### 📊 WAP Sinyal Dağılımı (Bina Bazında)")
        secili_wap = st.selectbox("WAP Seçin", top_names[:20],
                                  format_func=lambda x: f"{x} (Önem: {importance[wap_cols.index(x)]})")

        fig_dist = go.Figure()
        for bina in [0, 1, 2]:
            vals = train[train["BUILDINGID"] == bina][secili_wap]
            vals = vals[vals != 100]
            if len(vals) > 0:
                fig_dist.add_trace(go.Histogram(
                    x=vals, name=BINA_ISIMLERI[bina],
                    marker_color=BINA_RENKLERI[bina], opacity=0.7,
                    nbinsx=30
                ))
        fig_dist.update_layout(
            barmode="overlay", xaxis_title="Sinyal Gücü (dBm)",
            yaxis_title="Frekans", height=300,
            legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center")
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        # Kat bazında dağılım
        st.markdown(f"#### 📊 {secili_wap} - Kat Bazında Dağılım")
        fig_floor = go.Figure()
        renk_kat = ["#3498DB", "#2ECC71", "#E67E22", "#9B59B6", "#E74C3C"]
        for kat in range(5):
            vals = train[train["FLOOR"] == kat][secili_wap]
            vals = vals[vals != 100]
            if len(vals) > 0:
                fig_floor.add_trace(go.Box(
                    y=vals, name=f"Kat {kat}",
                    marker_color=renk_kat[kat],
                    boxpoints="outliers"
                ))
        fig_floor.update_layout(yaxis_title="Sinyal Gücü (dBm)", height=300)
        st.plotly_chart(fig_floor, use_container_width=True)


# ============================================================
# TAB 4: NASIL ÇALIŞIYOR?
# ============================================================
with tab4:
    st.markdown("### 🧠 LightGBM Nasıl Çalışıyor?")

    st.markdown("""
    ---
    #### 📱 Adım 1: Veri Toplama
    Telefonunuz etraftaki **520 WiFi Access Point**'ten sinyal gücü (RSSI) ölçer.
    Her ölçüm `-104` ile `0` dBm arasında bir değerdir. `0` = çok güçlü, `-104` = çok zayıf.

    > 💡 **Gerçek hayat:** Telefonunuz her an onlarca WiFi sinyali algılar. Bu sinyallerin
    > kombinasyonu her konum için benzersiz bir **"parmak izi"** oluşturur.
    ---
    """)

    st.markdown("""
    #### 🌳 Adım 2: LightGBM (Gradient Boosting)
    """)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **Random Forest (Eski Yöntem):**
        - 500 ağaç **bağımsız** eğitilir
        - Her ağaç kendi başına tahmin yapar
        - Sonuç: **çoğunluk oyu** ile karar
        - Basit ama sınırlı
        """)
    with col_b:
        st.markdown("""
        **LightGBM (Bu Model):**
        - 500 ağaç **sıralı** eğitilir
        - Her ağaç öncekinin **hatasını düzeltir**
        - Sonuç: tüm ağaçların katkısının **toplamı**
        - Daha akıllı ve güçlü
        """)

    st.markdown("---")

    st.markdown("""
    #### 🔄 Gradient Boosting Adım Adım

    ```
    Ağaç 1: "WAP248 sinyali > -60 ise → muhtemelen Bina 2"
             ↓ (hata: bazı Bina 1 örnekleri yanlış)

    Ağaç 2: "WAP501 sinyali > -70 ise → Bina 1'dir düzelt"
             ↓ (kalan hata: kat tahminleri)

    Ağaç 3: "WAP035 sinyali > -80 ise → Kat 2'dir düzelt"
             ↓ ...

    Ağaç 500: Son ince ayarlar
             ↓

    Final: Tüm 500 ağacın tahminlerini topla → Bina 2, Kat 3
    ```
    ---
    """)

    st.markdown("""
    #### 📊 Dataset Bilgisi

    | Özellik | Değer |
    |---------|-------|
    | **Kaynak** | UJIndoorLoc (İspanya üniversite kampüsü) |
    | **Eğitim verisi** | 19,937 ölçüm |
    | **Test verisi** | 1,111 ölçüm |
    | **WiFi AP sayısı** | 520 |
    | **Bina sayısı** | 3 |
    | **Kat sayısı** | 5 (0-4) |
    | **Veri seyrekliği** | %96.5 (çoğu WAP sinyal algılamıyor) |
    """)

    st.markdown("---")

    st.markdown("""
    #### 🌍 Gündelik Hayatta Kullanım Alanları

    | Alan | Uygulama |
    |------|----------|
    | 🏥 **Hastane** | Hasta/doktor konum takibi, acil yönlendirme |
    | 🛒 **AVM** | Mağaza içi navigasyon, müşteri analizi |
    | 🏭 **Fabrika** | Ekipman/personel takibi, güvenlik |
    | 🏢 **Ofis** | Toplantı odası doluluk, akıllı HVAC |
    | ✈️ **Havalimanı** | Yolcu yönlendirme, gate navigasyonu |
    | 🏫 **Üniversite** | Öğrenci yoğunluk analizi, kampüs navigasyonu |
    """)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 📡 WiFi Konum Tespiti")
    st.markdown("---")
    st.markdown(f"""
    **Model:** LightGBM
    **Eğitim verisi:** {len(train):,} ölçüm
    **Test verisi:** {len(test):,} ölçüm
    **WAP sayısı:** {len(wap_cols)}
    **Doğruluk:** %{accuracy_score(y_test, y_pred)*100:.1f}
    """)
    st.markdown("---")
    st.markdown("""
    **Teknolojiler:**
    `Python` `LightGBM` `Streamlit` `Plotly`
    """)
    st.markdown("---")
    st.markdown("Berke Baran Tozkoparan")
