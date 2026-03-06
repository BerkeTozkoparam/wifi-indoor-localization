import streamlit as st
import folium
from streamlit_folium import st_folium
import json
import os
import subprocess
import base64
import uuid
from datetime import datetime

# ── Sayfa Ayarları ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kadın Güvenlik Haritası | 8 Mart",
    page_icon="💜",
    layout="wide",
)

# ── Sabitler ────────────────────────────────────────────────────────────────
MARKERS_FILE  = os.path.join(os.path.dirname(__file__), "markers.json")
UPLOADS_DIR   = os.path.join(os.path.dirname(__file__), "uploads")
BURSA_CENTER  = [40.1885, 29.0610]
VALIDATOR_BIN = os.path.join(os.path.dirname(__file__), "validate")

os.makedirs(UPLOADS_DIR, exist_ok=True)

DURUM_RENK = {
    "Güvenli":   {"color": "green",  "icon": "check",       "hex": "#2ecc71", "emoji": "🟢"},
    "Dikkatli":  {"color": "orange", "icon": "exclamation", "hex": "#f39c12", "emoji": "🟡"},
    "Tehlikeli": {"color": "red",    "icon": "times",       "hex": "#e74c3c", "emoji": "🔴"},
}

KATEGORILER = [
    "Aydınlatma Sorunu",
    "Kalabalık / Güvenli Alan",
    "Yalnız Yürüme",
    "Taciz / Rahatsızlık",
    "Güvenli Mekân (Kafe, AVM vb.)",
    "Polis / Güvenlik Noktası",
    "Diğer",
]

# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }

    .hero {
        background: linear-gradient(135deg, #6c3483 0%, #c0392b 50%, #e91e8c 100%);
        padding: 1.6rem 2.2rem;
        border-radius: 16px;
        margin-bottom: 1.2rem;
        color: white;
    }
    .hero h1 { font-size: 2rem; font-weight: 800; margin: 0; }
    .hero p  { font-size: 0.95rem; margin: 0.3rem 0 0; opacity: 0.9; }

    .stat-card {
        background: white;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        border-left: 5px solid;
    }
    .stat-num { font-size: 1.9rem; font-weight: 800; }
    .stat-lbl { font-size: 0.78rem; color: #666; margin-top: 2px; }

    .panel-title {
        font-size: 1rem; font-weight: 700;
        color: #4a235a; margin-bottom: 0.7rem;
        border-bottom: 2px solid #e8c7f0;
        padding-bottom: 0.35rem;
    }

    .marker-card {
        background: #fdf6ff;
        border-radius: 10px;
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid;
        font-size: 0.86rem;
        position: relative;
    }
    .mc-title { font-weight: 700; font-size: 0.92rem; }
    .mc-meta  { color: #999; margin-top: 1px; font-size: 0.8rem; }

    div[data-testid="stForm"] { background: #fdf6ff; border-radius: 14px; padding: 1rem; }

    .stButton > button {
        background: linear-gradient(135deg, #6c3483, #e91e8c);
        color: white; border: none; border-radius: 10px;
        font-weight: 700; font-size: 0.93rem; width: 100%; padding: 0.55rem;
        transition: opacity .2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Temizle butonu farklı renk */
    .btn-danger > button {
        background: #e74c3c !important;
    }

    .stSelectbox label, .stTextArea label, .stTextInput label {
        font-weight: 600; color: #4a235a;
    }

    /* Harita bilgi balonu */
    .konum-badge {
        background: linear-gradient(135deg,#6c3483,#e91e8c);
        color: white; border-radius: 10px; padding: 0.5rem 1rem;
        font-size: 0.88rem; font-weight: 600; margin-top: 0.5rem;
        display: inline-block;
    }

    .legend-box {
        background: white; border-radius: 10px;
        padding: 0.6rem 1rem; margin-bottom: 0.8rem;
        box-shadow: 0 1px 8px rgba(0,0,0,0.07);
        font-size: 0.85rem;
    }

    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Veri Yönetimi ────────────────────────────────────────────────────────────
def yukle_markers():
    if not os.path.exists(MARKERS_FILE):
        return []
    try:
        with open(MARKERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def kaydet_markers(markers: list):
    with open(MARKERS_FILE, "w", encoding="utf-8") as f:
        json.dump(markers, f, ensure_ascii=False, indent=2)

def fotograf_kaydet(uploaded_file) -> str | None:
    """Yüklenen fotoğrafı uploads/ klasörüne kaydeder, dosya adını döner."""
    if uploaded_file is None:
        return None
    ext = os.path.splitext(uploaded_file.name)[-1].lower()
    dosya_adi = f"{uuid.uuid4().hex}{ext}"
    yol = os.path.join(UPLOADS_DIR, dosya_adi)
    with open(yol, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dosya_adi

def fotograf_base64(dosya_adi: str) -> str | None:
    """Kaydedilmiş fotoğrafı base64 string'e çevirir (popup için)."""
    if not dosya_adi:
        return None
    yol = os.path.join(UPLOADS_DIR, dosya_adi)
    if not os.path.exists(yol):
        return None
    ext = os.path.splitext(dosya_adi)[-1].lstrip(".").lower()
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    with open(yol, "rb") as f:
        return f"data:image/{mime};base64,{base64.b64encode(f.read()).decode()}"

def c_dogrula_sessiz(markers: list) -> bool:
    """Kayıt anında sessizce çalışır; sadece False döner, UI'a yansımaz."""
    if not os.path.exists(VALIDATOR_BIN):
        return all(
            all(k in m for k in ("lat", "lon", "durum", "tarih")) and
            m["durum"] in DURUM_RENK
            for m in markers
        )
    try:
        tmp = "/tmp/validate_tmp.json"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(markers, f, ensure_ascii=False)
        r = subprocess.run([VALIDATOR_BIN, tmp], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False

# ── Harita ──────────────────────────────────────────────────────────────────
def harita_olustur(markers: list, filtre: str = "Tümü") -> folium.Map:
    m = folium.Map(location=BURSA_CENTER, zoom_start=13, tiles="CartoDB positron")

    gosterilecek = [mk for mk in markers if filtre == "Tümü" or mk["durum"] == filtre]

    for mk in gosterilecek:
        ri = DURUM_RENK.get(mk["durum"], {"color": "gray", "icon": "info-sign", "hex": "#888"})
        not_str = mk.get("not", "") or "—"
        img_tag = ""
        b64 = fotograf_base64(mk.get("fotograf"))
        if b64:
            img_tag = f"<img src='{b64}' style='width:100%;border-radius:6px;margin:5px 0'>"
        popup_html = f"""
        <div style='font-family:Nunito,sans-serif;min-width:200px;padding:4px'>
          <b style='color:{ri["hex"]};font-size:1.05rem'>{mk["durum"]}</b><br>
          <span style='color:#555;font-size:0.85rem'>📂 {mk.get("kategori","—")}</span><br>
          <span style='color:#333;font-size:0.88rem'>📝 {not_str}</span>
          {img_tag}
          <hr style='margin:5px 0;border-color:#eee'>
          <span style='color:#bbb;font-size:0.74rem'>📍 {mk["lat"]:.4f}, {mk["lon"]:.4f}</span><br>
          <span style='color:#bbb;font-size:0.74rem'>🕐 {mk["tarih"]}</span>
        </div>
        """
        folium.Marker(
            location=[mk["lat"], mk["lon"]],
            popup=folium.Popup(popup_html, max_width=240),
            tooltip=f'{ri.get("emoji","")} {mk["durum"]} — {mk.get("kategori","")[:22]}',
            icon=folium.Icon(color=ri["color"], icon=ri["icon"], prefix="fa"),
        ).add_to(m)

    return m

# ── Session State ────────────────────────────────────────────────────────────
if "markers" not in st.session_state:
    st.session_state.markers = yukle_markers()
if "secilen_konum" not in st.session_state:
    st.session_state.secilen_konum = None
if "filtre" not in st.session_state:
    st.session_state.filtre = "Tümü"
if "silme_onay" not in st.session_state:
    st.session_state.silme_onay = False

# ── Başlık ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>Kadın Güvenlik Haritası</h1>
  <p>8 Mart Dünya Kadınlar Günü &nbsp;|&nbsp; Bursa &nbsp;•&nbsp;
     Haritaya tıkla, bölgeni işaretle. Birlikte daha güvenli bir şehir.</p>
</div>
""", unsafe_allow_html=True)

# ── İstatistik Kartları ──────────────────────────────────────────────────────
markers = st.session_state.markers
g = sum(1 for m in markers if m["durum"] == "Güvenli")
d = sum(1 for m in markers if m["durum"] == "Dikkatli")
t = sum(1 for m in markers if m["durum"] == "Tehlikeli")

c1, c2, c3, c4 = st.columns(4)
for col, num, lbl, renk in [
    (c1, len(markers), "Toplam İşaret", "#9b59b6"),
    (c2, g,           "Güvenli",       "#2ecc71"),
    (c3, d,           "Dikkatli",      "#f39c12"),
    (c4, t,           "Tehlikeli",     "#e74c3c"),
]:
    with col:
        st.markdown(f"""<div class="stat-card" style="border-color:{renk}">
            <div class="stat-num" style="color:{renk}">{num}</div>
            <div class="stat-lbl">{lbl}</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Ana Layout ──────────────────────────────────────────────────────────────
col_harita, col_panel = st.columns([3, 1.15], gap="medium")

# ════════════════════════════════════════════════════════════ HARITA
with col_harita:
    # Filtre satırı
    f_col1, f_col2 = st.columns([1, 3])
    with f_col1:
        st.markdown('<div style="padding-top:6px;font-weight:700;color:#4a235a">Filtrele:</div>',
                    unsafe_allow_html=True)
    with f_col2:
        filtre_sec = st.radio(
            "filtre",
            ["Tümü", "Güvenli", "Dikkatli", "Tehlikeli"],
            horizontal=True,
            label_visibility="collapsed",
            key="filtre_radio",
        )
        st.session_state.filtre = filtre_sec

    harita = harita_olustur(st.session_state.markers, st.session_state.filtre)
    map_data = st_folium(harita, height=530, use_container_width=True, key="ana_harita")

    # Tıklanan konum yakala
    if map_data and map_data.get("last_clicked"):
        latlng = map_data["last_clicked"]
        st.session_state.secilen_konum = (latlng["lat"], latlng["lng"])

    if st.session_state.secilen_konum:
        lat, lon = st.session_state.secilen_konum
        st.markdown(
            f'<div class="konum-badge">📍 Seçilen: {lat:.5f}, {lon:.5f} &nbsp;—&nbsp; Sağ panelden bilgileri gir ve ekle</div>',
            unsafe_allow_html=True,
        )

# ════════════════════════════════════════════════════════════ PANEL
with col_panel:

    # ── Bölge Ekle ──────────────────────────────────────────────────────
    st.markdown('<div class="panel-title">Bölge Ekle</div>', unsafe_allow_html=True)

    with st.form("marker_form", clear_on_submit=True):
        durum = st.selectbox("Durum", list(DURUM_RENK.keys()))
        kategori = st.selectbox("Kategori", KATEGORILER)
        not_txt = st.text_area(
            "Not (isteğe bağlı)", height=70,
            placeholder="Bu bölge hakkında kısa bir not..."
        )
        fotograf = st.file_uploader(
            "Fotoğraf ekle (isteğe bağlı)",
            type=["jpg", "jpeg", "png", "webp"],
            help="Bölgeyi belgeleyen bir fotoğraf yükleyebilirsin.",
        )
        submitted = st.form_submit_button("Haritaya Ekle")

        if submitted:
            if st.session_state.secilen_konum is None:
                st.warning("Önce haritada bir noktaya tıkla!")
            else:
                lat, lon = st.session_state.secilen_konum
                dosya_adi = fotograf_kaydet(fotograf)
                yeni = {
                    "lat":      lat,
                    "lon":      lon,
                    "durum":    durum,
                    "kategori": kategori,
                    "not":      not_txt.strip(),
                    "fotograf": dosya_adi,
                    "tarih":    datetime.now().strftime("%d.%m.%Y %H:%M"),
                }
                st.session_state.markers.append(yeni)
                kaydet_markers(st.session_state.markers)
                c_dogrula_sessiz(st.session_state.markers)   # arka planda C doğrulama
                st.session_state.secilen_konum = None
                st.success("Bölge eklendi!")
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── İşaretler Listesi ─────────────────────────────────────────────────
    st.markdown('<div class="panel-title">Tüm İşaretler</div>', unsafe_allow_html=True)

    filtre_liste = st.session_state.filtre
    liste = [mk for mk in reversed(st.session_state.markers)
             if filtre_liste == "Tümü" or mk["durum"] == filtre_liste]

    if not liste:
        st.caption("Gösterilecek işaret yok.")
    else:
        # Scrollable container
        with st.container(height=340):
            for idx, mk in enumerate(liste):
                renk = DURUM_RENK.get(mk["durum"], {}).get("hex", "#888")
                emoji = DURUM_RENK.get(mk["durum"], {}).get("emoji", "")
                not_goster = mk.get("not") or "—"

                # Gerçek indeks (silebilmek için)
                gercek_idx = len(st.session_state.markers) - 1 - \
                    next(i for i, m in enumerate(reversed(st.session_state.markers))
                         if m is mk)

                thumb = ""
                b64 = fotograf_base64(mk.get("fotograf"))
                if b64:
                    thumb = f"<img src='{b64}' style='width:100%;border-radius:6px;margin-top:5px;max-height:110px;object-fit:cover'>"
                st.markdown(f"""
                <div class="marker-card" style="border-color:{renk}">
                  <div class="mc-title" style="color:{renk}">{emoji} {mk['durum']}</div>
                  <div style="color:#555">{mk.get('kategori','')}</div>
                  <div class="mc-meta">{not_goster}</div>
                  {thumb}
                  <div class="mc-meta">{mk['tarih']}</div>
                </div>
                """, unsafe_allow_html=True)

                if st.button("Sil", key=f"sil_{id(mk)}_{idx}"):
                    # Fotoğrafı diskten sil
                    if mk.get("fotograf"):
                        try:
                            os.remove(os.path.join(UPLOADS_DIR, mk["fotograf"]))
                        except FileNotFoundError:
                            pass
                    st.session_state.markers = [
                        m for m in st.session_state.markers if m is not mk
                    ]
                    kaydet_markers(st.session_state.markers)
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tüm Verileri Temizle ──────────────────────────────────────────────
    if not st.session_state.silme_onay:
        if st.button("Tüm Verileri Temizle"):
            st.session_state.silme_onay = True
            st.rerun()
    else:
        st.warning("Tüm işaretler silinecek. Emin misin?")
        col_e, col_h = st.columns(2)
        with col_e:
            if st.button("Evet, Sil"):
                # Tüm fotoğrafları sil
                for mk in st.session_state.markers:
                    if mk.get("fotograf"):
                        try:
                            os.remove(os.path.join(UPLOADS_DIR, mk["fotograf"]))
                        except FileNotFoundError:
                            pass
                st.session_state.markers = []
                kaydet_markers([])
                st.session_state.silme_onay = False
                st.rerun()
        with col_h:
            if st.button("Hayır"):
                st.session_state.silme_onay = False
                st.rerun()

# ── Alt Bilgi ────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="margin-top:2rem;border-color:#e8c7f0">
<div style="text-align:center;color:#9b59b6;font-size:0.82rem;padding-bottom:1rem">
  8 Mart Dünya Kadınlar Günü &nbsp;•&nbsp; Güvenli bir şehir hakkımız
</div>
""", unsafe_allow_html=True)
