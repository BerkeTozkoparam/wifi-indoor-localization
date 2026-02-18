"""
WiFi Fingerprinting - İnteraktif Bina & Kat Tespiti Demo
=========================================================
Çalıştır: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, confusion_matrix
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

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem; border-radius: 12px; color: white;
        text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card h2 { margin: 0; font-size: 2.2rem; }
    .metric-card p { margin: 0.3rem 0 0 0; opacity: 0.9; font-size: 0.9rem; }
    .success-card { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .warning-card { background: linear-gradient(135deg, #F2994A 0%, #F2C94C 100%); }
    .info-card { background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%); }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HAZIR MODEL VE VERİ YÜKLE (anlık, eğitim yok)
# ============================================================
@st.cache_resource
def load_all():
    model = joblib.load("model.pkl")
    le = joblib.load("label_encoder.pkl")

    results = np.load("test_results.npz")
    X_test = results["X_test"]
    y_test = results["y_test"]
    y_pred = results["y_pred"]
    test_building = results["test_building"]
    test_floor = results["test_floor"]

    sample_data = pd.read_csv("sample_data.csv")

    with open("wap_cols.json") as f:
        wap_cols = json.load(f)

    importance = np.load("feature_importance.npy")

    return model, le, X_test, y_test, y_pred, test_building, test_floor, sample_data, wap_cols, importance


model, le, X_test, y_test, y_pred, test_building, test_floor, sample_data, wap_cols, importance = load_all()

# Sabitler
BINA_ISIMLERI = {0: "Muhendislik Fakultesi", 1: "Fen Fakultesi", 2: "Kutuphane"}
BINA_KATLARI = {0: [0, 1, 2, 3], 1: [0, 1, 2, 3], 2: [0, 1, 2, 3, 4]}
BINA_RENKLERI = {0: "#4A90D9", 1: "#E67E22", 2: "#2ECC71"}

# ============================================================
# BASLIK
# ============================================================
st.markdown("""
<div style="text-align:center; padding:1rem 0;">
    <h1>📡 WiFi Fingerprinting ile Bina & Kat Tespiti</h1>
    <p style="font-size:1.1rem; color:#666;">
        520 WiFi Access Point sinyalinden konum tahmin eden LightGBM modeli
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🏢 Canli Simulasyon", "📊 Model Performansi",
    "🔬 Sinyal Analizi", "🧠 Nasil Calisiyor?"
])

# ============================================================
# TAB 1: CANLI SİMÜLASYON
# ============================================================
with tab1:
    st.markdown("### 🎮 Bina Icinde Yuruyus Simulasyonu")
    st.markdown("Bir bina ve kat secin, model WiFi sinyallerinden konumunuzu tahmin etsin!")

    col_control, col_viz = st.columns([1, 2.5])

    with col_control:
        st.markdown("#### ⚙️ Kontroller")

        secili_bina = st.selectbox(
            "🏢 Bina Secin", options=[0, 1, 2],
            format_func=lambda x: f"Bina {x} - {BINA_ISIMLERI[x]}"
        )
        secili_kat = st.selectbox(
            "🏗️ Gercek Katiniz", options=BINA_KATLARI[secili_bina],
            format_func=lambda x: f"Kat {x}"
        )
        gurultu = st.slider(
            "📶 Sinyal Gurultusu (dBm)", min_value=0, max_value=20, value=5,
            help="Gercek hayattaki sinyal dalgalanmasi. 0=ideal, 20=cok gurultulu"
        )
        kisi_sayisi = st.slider("👥 Simule Edilecek Kisi", min_value=1, max_value=10, value=5)
        test_et = st.button("🚀 Simulasyonu Baslat", use_container_width=True, type="primary")

    with col_viz:
        if test_et:
            mask = (sample_data["BUILDINGID"] == secili_bina) & (sample_data["FLOOR"] == secili_kat)
            ornekler = sample_data[mask]

            if len(ornekler) < kisi_sayisi:
                st.error("Bu bina+kat icin yeterli veri yok!")
            else:
                np.random.seed(None)
                secilen = ornekler.sample(n=kisi_sayisi)

                sonuclar = []
                for _, row in secilen.iterrows():
                    sinyal = row[wap_cols].values.astype(float).copy()
                    algilanan = sinyal != 100

                    if gurultu > 0:
                        noise = np.random.normal(0, gurultu, size=sinyal.shape)
                        sinyal[algilanan] = np.clip(sinyal[algilanan] + noise[algilanan], -104, 0)

                    sinyal_model = np.where(sinyal == 100, -105, sinyal).reshape(1, -1)
                    tahmin = model.predict(sinyal_model)[0]
                    proba = model.predict_proba(sinyal_model)[0]
                    tahmin_label = le.inverse_transform([tahmin])[0]
                    t_bina, t_kat = int(tahmin_label.split("_")[0]), int(tahmin_label.split("_")[1])

                    sonuclar.append({
                        "tahmin_bina": t_bina, "tahmin_kat": t_kat,
                        "dogru": (t_bina == secili_bina) and (t_kat == secili_kat),
                        "guven": float(proba.max()) * 100,
                        "aktif_wap": int(algilanan.sum()),
                    })

                # Bina gorseli
                fig = go.Figure()
                kat_sayisi = len(BINA_KATLARI[secili_bina])
                bina_w, kat_h = 6, 1.2

                for k in range(kat_sayisi):
                    y0 = k * kat_h
                    renk = BINA_RENKLERI[secili_bina]
                    opacity = 0.25 if k == secili_kat else 0.06
                    line_w = 3 if k == secili_kat else 1

                    fig.add_shape(type="rect", x0=0, y0=y0, x1=bina_w, y1=y0 + kat_h,
                        fillcolor=renk, opacity=opacity, line=dict(color=renk, width=line_w))
                    fig.add_annotation(x=-0.4, y=y0 + kat_h / 2,
                        text=f"<b>Kat {k}</b>", showarrow=False, font=dict(size=13, color="#333"))

                y_top = kat_sayisi * kat_h
                fig.add_shape(type="line", x0=0, y0=y_top, x1=bina_w, y1=y_top,
                    line=dict(color="gray", width=2))

                for k in range(kat_sayisi):
                    fig.add_trace(go.Scatter(
                        x=[bina_w - 0.3], y=[k * kat_h + kat_h * 0.75],
                        mode="markers+text", text=["📡"], textposition="middle center",
                        textfont=dict(size=16), marker=dict(size=1, color="rgba(0,0,0,0)"),
                        showlegend=False, hoverinfo="skip"))

                dogru_x, dogru_y, dogru_text = [], [], []
                yanlis_x, yanlis_y, yanlis_text = [], [], []

                for i, s in enumerate(sonuclar):
                    x_pos = (i + 1) * bina_w / (kisi_sayisi + 1)
                    y_pos = secili_kat * kat_h + kat_h * 0.45
                    hover = (f"Kisi {i+1}<br>Tahmin: Bina {s['tahmin_bina']} Kat {s['tahmin_kat']}<br>"
                             f"Guven: %{s['guven']:.0f}<br>Aktif WAP: {s['aktif_wap']}")
                    if s["dogru"]:
                        dogru_x.append(x_pos); dogru_y.append(y_pos); dogru_text.append(hover)
                    else:
                        yanlis_x.append(x_pos); yanlis_y.append(y_pos); yanlis_text.append(hover)

                if dogru_x:
                    fig.add_trace(go.Scatter(x=dogru_x, y=dogru_y, mode="markers",
                        marker=dict(size=28, color="#27AE60", symbol="circle",
                                    line=dict(width=2, color="white")),
                        name=f"✅ Dogru ({len(dogru_x)})", text=dogru_text, hoverinfo="text"))
                if yanlis_x:
                    fig.add_trace(go.Scatter(x=yanlis_x, y=yanlis_y, mode="markers",
                        marker=dict(size=28, color="#E74C3C", symbol="x",
                                    line=dict(width=3, color="white")),
                        name=f"❌ Yanlis ({len(yanlis_x)})", text=yanlis_text, hoverinfo="text"))

                fig.update_layout(
                    title=dict(text=f"<b>{BINA_ISIMLERI[secili_bina]}</b> - Kat {secili_kat} Simulasyonu",
                               font=dict(size=18)),
                    xaxis=dict(visible=False, range=[-1, bina_w + 1]),
                    yaxis=dict(visible=False, range=[-0.5, kat_sayisi * kat_h + 0.8], scaleanchor="x"),
                    height=450,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.15, x=0.5, xanchor="center",
                                font=dict(size=14)),
                    margin=dict(l=40, r=20, t=60, b=60), plot_bgcolor="white")
                st.plotly_chart(fig, use_container_width=True)

                dogru_sayi = sum(1 for s in sonuclar if s["dogru"])
                ort_guven = np.mean([s["guven"] for s in sonuclar])
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f'<div class="metric-card success-card"><h2>{dogru_sayi}/{kisi_sayisi}</h2><p>Dogru Tahmin</p></div>', unsafe_allow_html=True)
                with m2:
                    acc = dogru_sayi / kisi_sayisi * 100
                    cls = "success-card" if acc >= 80 else "warning-card"
                    st.markdown(f'<div class="metric-card {cls}"><h2>%{acc:.0f}</h2><p>Dogruluk Orani</p></div>', unsafe_allow_html=True)
                with m3:
                    st.markdown(f'<div class="metric-card info-card"><h2>%{ort_guven:.0f}</h2><p>Ortalama Guven</p></div>', unsafe_allow_html=True)

                st.markdown("#### 📋 Detayli Sonuclar")
                tablo = pd.DataFrame(sonuclar)
                tablo.index = [f"Kisi {i+1}" for i in range(len(tablo))]
                tablo.columns = ["Tahmin Bina", "Tahmin Kat", "Dogru?", "Guven %", "Aktif WAP"]
                tablo["Dogru?"] = tablo["Dogru?"].map({True: "✅", False: "❌"})
                tablo["Guven %"] = tablo["Guven %"].apply(lambda x: f"%{x:.1f}")
                st.dataframe(tablo, use_container_width=True)
        else:
            st.info("👆 Soldaki kontrolleri ayarlayip **Simulasyonu Baslat** butonuna tiklayin!")

            fig = go.Figure()
            positions = {0: (1, 1), 1: (4, 1), 2: (2.5, 3.5)}
            sizes = {0: 4, 1: 4, 2: 5}
            for bina_id, (x, y) in positions.items():
                fig.add_trace(go.Scatter(
                    x=[x], y=[y], mode="markers+text",
                    marker=dict(size=60 + sizes[bina_id] * 8, color=BINA_RENKLERI[bina_id],
                                opacity=0.3, line=dict(width=3, color=BINA_RENKLERI[bina_id])),
                    text=[f"🏢<br><b>{BINA_ISIMLERI[bina_id]}</b><br>{sizes[bina_id]} kat"],
                    textposition="middle center", textfont=dict(size=11),
                    showlegend=False, hoverinfo="skip"))
            fig.update_layout(title="<b>🏫 Kampus Haritasi</b>",
                xaxis=dict(visible=False, range=[-0.5, 5.5]),
                yaxis=dict(visible=False, range=[-0.5, 5.5], scaleanchor="x"),
                height=400, plot_bgcolor="#F8F9FA", margin=dict(l=20, r=20, t=50, b=20))
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
    t_building = test_building
    t_floor = test_floor

    building_acc = accuracy_score(t_building, p_building) * 100
    floor_acc = accuracy_score(t_floor, p_floor) * 100
    overall_acc = accuracy_score(y_test, y_pred) * 100

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-card"><h2>%{building_acc:.1f}</h2><p>🏢 Bina Dogrulugu</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card success-card"><h2>%{floor_acc:.1f}</h2><p>🏗️ Kat Dogrulugu</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card info-card"><h2>%{overall_acc:.1f}</h2><p>📍 Genel Dogruluk</p></div>', unsafe_allow_html=True)

    st.markdown("")
    col_cm1, col_cm2 = st.columns(2)

    with col_cm1:
        cm_b = confusion_matrix(t_building, p_building)
        fig_b = go.Figure(data=go.Heatmap(
            z=cm_b, x=["Bina 0", "Bina 1", "Bina 2"], y=["Bina 0", "Bina 1", "Bina 2"],
            colorscale="Blues", texttemplate="%{z}", textfont=dict(size=16),
            hovertemplate="Gercek: %{y}<br>Tahmin: %{x}<br>Sayi: %{z}<extra></extra>"))
        fig_b.update_layout(title="<b>Bina Confusion Matrix</b>",
            xaxis_title="Tahmin", yaxis_title="Gercek", height=380, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_b, use_container_width=True)

    with col_cm2:
        cm_f = confusion_matrix(t_floor, p_floor)
        kat_labels = [f"Kat {i}" for i in range(5)]
        fig_f = go.Figure(data=go.Heatmap(
            z=cm_f, x=kat_labels, y=kat_labels, colorscale="Greens",
            texttemplate="%{z}", textfont=dict(size=16),
            hovertemplate="Gercek: %{y}<br>Tahmin: %{x}<br>Sayi: %{z}<extra></extra>"))
        fig_f.update_layout(title="<b>Kat Confusion Matrix</b>",
            xaxis_title="Tahmin", yaxis_title="Gercek", height=380, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_f, use_container_width=True)

    st.markdown("#### 🏢 Bina Bazinda Kat Dogruluklari")
    fig_bar = go.Figure()
    for bina in [0, 1, 2]:
        b_mask = t_building == bina
        floors = sorted(np.unique(t_floor[b_mask]))
        accs = []
        for f in floors:
            f_mask = b_mask & (t_floor == f)
            accs.append((p_floor[f_mask] == f).mean() * 100 if f_mask.sum() > 0 else 0)
        fig_bar.add_trace(go.Bar(
            x=[f"Kat {f}" for f in floors], y=accs, name=BINA_ISIMLERI[bina],
            marker_color=BINA_RENKLERI[bina], text=[f"%{a:.0f}" for a in accs], textposition="outside"))
    fig_bar.update_layout(barmode="group", yaxis_range=[0, 110], yaxis_title="Dogruluk (%)",
        height=400, legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"))
    st.plotly_chart(fig_bar, use_container_width=True)


# ============================================================
# TAB 3: SİNYAL ANALİZİ
# ============================================================
with tab3:
    st.markdown("### 🔬 WiFi Sinyal Analizi")
    col_s1, col_s2 = st.columns([1, 1])

    with col_s1:
        st.markdown("#### 📡 En Onemli 20 WiFi Access Point")
        top_idx = np.argsort(importance)[-20:][::-1]
        top_names = [wap_cols[i] for i in top_idx]
        top_vals = importance[top_idx]

        fig_imp = go.Figure(go.Bar(
            x=top_vals[::-1], y=top_names[::-1], orientation="h",
            marker=dict(color=top_vals[::-1], colorscale="YlOrRd"),
            hovertemplate="%{y}: %{x} kullanim<extra></extra>"))
        fig_imp.update_layout(height=500, xaxis_title="Onem Skoru", margin=dict(l=80))
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_s2:
        st.markdown("#### 📊 WAP Sinyal Dagilimi (Bina Bazinda)")
        secili_wap = st.selectbox("WAP Secin", top_names[:20])

        fig_dist = go.Figure()
        for bina in [0, 1, 2]:
            vals = sample_data[sample_data["BUILDINGID"] == bina][secili_wap]
            vals = vals[vals != 100]
            if len(vals) > 0:
                fig_dist.add_trace(go.Histogram(
                    x=vals, name=BINA_ISIMLERI[bina], marker_color=BINA_RENKLERI[bina],
                    opacity=0.7, nbinsx=30))
        fig_dist.update_layout(barmode="overlay", xaxis_title="Sinyal Gucu (dBm)",
            yaxis_title="Frekans", height=300,
            legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"))
        st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown(f"#### 📊 {secili_wap} - Kat Bazinda Dagilim")
        fig_floor = go.Figure()
        renk_kat = ["#3498DB", "#2ECC71", "#E67E22", "#9B59B6", "#E74C3C"]
        for kat in range(5):
            vals = sample_data[sample_data["FLOOR"] == kat][secili_wap]
            vals = vals[vals != 100]
            if len(vals) > 0:
                fig_floor.add_trace(go.Box(y=vals, name=f"Kat {kat}", marker_color=renk_kat[kat]))
        fig_floor.update_layout(yaxis_title="Sinyal Gucu (dBm)", height=300)
        st.plotly_chart(fig_floor, use_container_width=True)


# ============================================================
# TAB 4: NASIL CALISIYOR?
# ============================================================
with tab4:
    st.markdown("""
    ### 🧠 LightGBM Nasil Calisiyor?
    ---
    #### 📱 Adim 1: Veri Toplama
    Telefonunuz etraftaki **520 WiFi Access Point**'ten sinyal gucu (RSSI) olcer.
    Her olcum `-104` ile `0` dBm arasinda bir degerdir. `0` = cok guclu, `-104` = cok zayif.

    > 💡 **Gercek hayat:** Telefonunuz her an onlarca WiFi sinyali algilar. Bu sinyallerin
    > kombinasyonu her konum icin benzersiz bir **"parmak izi"** olusturur.
    ---
    #### 🌳 Adim 2: LightGBM (Gradient Boosting)
    """)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **Random Forest (Eski Yontem):**
        - 500 agac **bagimsiz** egitilir
        - Her agac kendi basina tahmin yapar
        - Sonuc: **cogunluk oyu** ile karar
        """)
    with col_b:
        st.markdown("""
        **LightGBM (Bu Model):**
        - 500 agac **sirali** egitilir
        - Her agac oncekinin **hatasini duzeltir**
        - Sonuc: tum agaclarin katkisinin **toplami**
        """)

    st.markdown("""
    ---
    #### 🔄 Gradient Boosting Adim Adim
    ```
    Agac 1: "WAP248 sinyali > -60 ise → muhtemelen Bina 2"
             ↓ (hata: bazi Bina 1 ornekleri yanlis)
    Agac 2: "WAP501 sinyali > -70 ise → Bina 1'dir duzelt"
             ↓ (kalan hata: kat tahminleri)
    Agac 3: "WAP035 sinyali > -80 ise → Kat 2'dir duzelt"
             ↓ ...
    Agac 500: Son ince ayarlar
             ↓
    Final: Tum 500 agacin tahminlerini topla → Bina 2, Kat 3
    ```
    ---
    #### 🌍 Gundelik Hayatta Kullanim Alanlari

    | Alan | Uygulama |
    |------|----------|
    | 🏥 **Hastane** | Hasta/doktor konum takibi, acil yonlendirme |
    | 🛒 **AVM** | Magaza ici navigasyon, musteri analizi |
    | 🏭 **Fabrika** | Ekipman/personel takibi, guvenlik |
    | 🏢 **Ofis** | Toplanti odasi doluluk, akilli HVAC |
    | ✈️ **Havalimani** | Yolcu yonlendirme, gate navigasyonu |
    | 🏫 **Universite** | Ogrenci yogunluk analizi, kampus navigasyonu |
    """)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 📡 WiFi Konum Tespiti")
    st.markdown("---")
    st.markdown(f"""
    **Model:** LightGBM
    **Egitim verisi:** 19,937 olcum
    **Test verisi:** 1,111 olcum
    **WAP sayisi:** {len(wap_cols)}
    **Dogruluk:** %{accuracy_score(y_test, y_pred)*100:.1f}
    """)
    st.markdown("---")
    st.markdown("**Teknolojiler:**  \n`Python` `LightGBM` `Streamlit` `Plotly`")
    st.markdown("---")
    st.markdown("Berke Baran Tozkoparan")
