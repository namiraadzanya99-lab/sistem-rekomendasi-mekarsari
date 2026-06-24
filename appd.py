import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from datetime import datetime
import os

# =========================
# PAGE CONFIG
# =========================
APP_TITLE = "Sistem Toko Pertanian dan Pakan Ternak Mekarsari"
FILE_MASTER = "master_produk.xlsx"
FILE_REKAP = "rekap_transaksi.xlsx"
LOGO_PATH = "logo_mekarsari.png"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM STYLE
# =========================
st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(180deg, #f8fbff 0%, #f7fafc 100%);
        }

        .block-container {
            padding-top: 1.3rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3, h4 {
            color: #0f172a;
        }

        .top-banner {
            padding: 1.15rem 1.25rem;
            border-radius: 24px;
            background: linear-gradient(90deg, #eff6ff 0%, #ffffff 100%);
            border: 1px solid #dbeafe;
            box-shadow: 0 10px 24px rgba(59,130,246,.08);
            margin-bottom: 1rem;
        }

        .stMetric {
            background: white;
            padding: 14px 16px;
            border-radius: 16px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
        }

        div[data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid #e2e8f0;
            background: white;
        }

        .stButton > button {
            border-radius: 12px;
            padding: 0.6rem 1rem;
            font-weight: 600;
            border: none;
            background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%);
            color: white;
            box-shadow: 0 10px 22px rgba(59, 130, 246, 0.18);
        }

        .stButton > button:hover {
            background: linear-gradient(90deg, #1d4ed8 0%, #2563eb 100%);
            color: white;
        }

        .stDownloadButton > button {
            border-radius: 12px;
            padding: 0.6rem 1rem;
            font-weight: 600;
            border: none;
            background: linear-gradient(90deg, #0f766e 0%, #14b8a6 100%);
            color: white;
            box-shadow: 0 10px 22px rgba(20, 184, 166, 0.18);
        }

        .stDownloadButton > button:hover {
            background: linear-gradient(90deg, #115e59 0%, #0f766e 100%);
            color: white;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #eff6ff 0%, #ffffff 65%, #f8fbff 100%);
            border-right: 1px solid #e5eef9;
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.15rem;
            padding-bottom: 1.15rem;
        }

        .sidebar-box {
            padding: 1.25rem 1.15rem;
            border-radius: 28px;
            background: rgba(255,255,255,.9);
            border: 1px solid rgba(148,163,184,.20);
            box-shadow: 0 18px 40px rgba(37,99,235,.08);
        }

        .sidebar-logo {
            display: flex;
            justify-content: center;
            margin-bottom: 1rem;
        }

        .sidebar-title {
            font-size: 1.6rem;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.05;
            text-align: center;
            margin-bottom: 0.45rem;
        }

        .sidebar-title span {
            color: #3b82f6;
        }

        .sidebar-subtitle {
            color: #64748b;
            font-size: 0.95rem;
            text-align: center;
            margin-bottom: 0.95rem;
            line-height: 1.5;
        }

        .sidebar-divider {
            height: 1px;
            background: #e2e8f0;
            margin: 1rem 0 1.05rem 0;
        }

        .sidebar-heading {
            font-size: 1.25rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.5rem;
        }

        div[data-testid="stRadio"] {
            margin-top: 0.2rem;
        }

        div[data-testid="stRadio"] label {
            border: 1px solid #e5e7eb;
            background: rgba(255,255,255,.96);
            padding: 0.82rem 0.95rem;
            margin: 0.45rem 0;
            border-radius: 18px;
            box-shadow: 0 8px 24px rgba(15,23,42,.05);
            transition: all 0.2s ease;
        }

        div[data-testid="stRadio"] label:hover {
            border-color: #93c5fd;
            transform: translateX(2px);
        }

        div[data-testid="stRadio"] label p {
            font-size: 1rem;
            font-weight: 600;
            color: #0f172a;
        }

        .welcome-card {
            padding: 1.35rem 1.4rem;
            border-radius: 24px;
            background: linear-gradient(90deg, #ffffff 0%, #f8fbff 100%);
            border: 1px solid #dbeafe;
            box-shadow: 0 12px 26px rgba(59,130,246,.08);
        }

        .welcome-title {
            font-size: 2rem;
            font-weight: 800;
            color: #0f172a;
            margin: 0;
            line-height: 1.2;
        }

        .welcome-subtitle {
            font-size: 1rem;
            color: #64748b;
            margin-top: 0.35rem;
            margin-bottom: 0;
            line-height: 1.5;
        }

        .mini-card {
            padding: 1rem;
            border-radius: 18px;
            background: white;
            border: 1px solid #e2e8f0;
            box-shadow: 0 6px 18px rgba(15,23,42,.04);
        }

        .section-card {
            padding: 1rem;
            border-radius: 18px;
            background: white;
            border: 1px solid #e2e8f0;
            box-shadow: 0 6px 18px rgba(15,23,42,.04);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# PASTIKAN FILE REKAP ADA
# =========================
if not os.path.exists(FILE_REKAP):
    pd.DataFrame(columns=["Kode", "Tanggal", "Produk", "Harga", "Jumlah", "Total"]).to_excel(FILE_REKAP, index=False)

if not os.path.exists(FILE_MASTER):
    st.error(f"File master tidak ditemukan: {FILE_MASTER}")
    st.stop()

# =========================
# UTILITIES
# =========================
def format_rupiah(angka):
    if pd.isna(angka):
        return "-"
    try:
        return f"Rp{int(float(angka)):,}".replace(",", ".")
    except Exception:
        return "-"

def clean_numeric(series):
    return pd.to_numeric(
        series.astype("string")
        .str.replace(r"[^0-9,.\-]", "", regex=True)
        .str.replace(",", "", regex=False),
        errors="coerce"
    )

def normalize_text(series):
    return series.astype("string").str.strip()

def split_targets(target_value):
    if pd.isna(target_value):
        return []
    return [t.strip().lower() for t in str(target_value).split(",") if t.strip()]

def get_target_options(data_cluster, kategori, sub_kategori):
    subset = data_cluster[
        (data_cluster["Kategori"] == kategori) &
        (data_cluster["Sub_Kategori"] == sub_kategori)
    ].copy()

    opsi = []
    for val in subset["Target"].dropna().unique():
        for item in str(val).split(","):
            item = item.strip().lower()
            if item and item not in opsi:
                opsi.append(item)

    return opsi if opsi else ["umum"]

def generate_kode(file_path):
    if not os.path.exists(file_path):
        return "T001"

    df = pd.read_excel(file_path)
    if df.empty or "Kode" not in df.columns:
        return "T001"

    kode_num = (
        df["Kode"].astype(str)
        .str.extract(r"(\d+)")[0]
        .astype(float)
    )

    if kode_num.dropna().empty:
        return "T001"

    last = int(kode_num.max())
    return f"T{last + 1:03d}"

def apply_plot_style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.25)

def show_page_header(title, subtitle):
    st.markdown(
        f"""
        <div class="top-banner">
            <h1 style="margin:0;">{title}</h1>
            <p style="margin:.25rem 0 0 0; color:#64748b;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    master = pd.read_excel(FILE_MASTER)
    transaksi = pd.read_excel(FILE_REKAP)

    master.columns = master.columns.str.strip()
    transaksi.columns = transaksi.columns.str.strip()

    for df in [master, transaksi]:
        if "Produk" not in df.columns:
            raise ValueError("Kolom 'Produk' harus ada di kedua file.")
        df["Produk"] = normalize_text(df["Produk"])

    for col in ["Kategori", "Sub Kategori", "Target"]:
        if col in master.columns:
            master[col] = normalize_text(master[col])

    if "Harga Jual" in master.columns:
        master["Harga Jual"] = clean_numeric(master["Harga Jual"])

    for col in ["Jumlah", "Harga", "Total"]:
        if col in transaksi.columns:
            transaksi[col] = clean_numeric(transaksi[col])

    if "Tanggal" in transaksi.columns:
        transaksi["Tanggal"] = pd.to_datetime(transaksi["Tanggal"], errors="coerce")

    master_selected = master[
        ["Produk", "Kategori", "Sub Kategori", "Target", "Harga Jual"]
    ].drop_duplicates(subset=["Produk"])

    data = pd.merge(transaksi, master_selected, on="Produk", how="left")
    data.columns = data.columns.str.strip().str.replace(" ", "_", regex=False)

    return master_selected, transaksi, data

# =========================
# K-MEANS
# =========================
@st.cache_data
def proses_kmeans(data_input, n_cluster=3):
    agg = (
        data_input.groupby("Produk", as_index=False)
        .agg(
            qty=("Jumlah", "sum"),
            frekuensi=("Produk", "size")
        )
    )
    # =========================
    # CAPPING OUTLIER
    # =========================
    
    if data_input["Kategori"].nunique() == 1 and data_input["Kategori"].iloc[0] == "Peternakan":
        # Dataset peternakan
        agg["qty"] = agg["qty"].clip(upper=500)
    else:
        # Dataset gabungan dan pertanian
        agg["qty"] = agg["qty"].clip(upper=1000)
        
    scaler = StandardScaler()
    scaled = scaler.fit_transform(agg[["qty", "frekuensi"]])

    inertia = []
    for k in range(1, 6):
        km_elbow = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_elbow.fit(scaled)
        inertia.append(km_elbow.inertia_)

    km = KMeans(n_clusters=n_cluster, random_state=42, n_init=10)
    agg["cluster"] = km.fit_predict(scaled)

    cluster_mean = agg.groupby("cluster")[["qty", "frekuensi"]].mean().sort_values("qty")
    urutan_cluster = cluster_mean.index.tolist()

    if n_cluster == 2:
        label_nama = ["Kurang Laris", "Laris"]
    elif n_cluster == 3:
        label_nama = ["Kurang Laris", "Cukup Laris", "Sangat Laris"]
    else:
        label_nama = [f"Cluster {i}" for i in range(n_cluster)]

    label_map = {}
    for i, c in enumerate(urutan_cluster):
        label_map[c] = label_nama[i] if i < len(label_nama) else f"Cluster {c}"

    agg["cluster_label"] = agg["cluster"].map(label_map)

    dbi = davies_bouldin_score(scaled, agg["cluster"])
    sil = silhouette_score(scaled, agg["cluster"])
    centroid_original = scaler.inverse_transform(km.cluster_centers_)

    return agg, inertia, dbi, sil, km, centroid_original

# =========================
# DATA PREP
# =========================
master, transaksi, data = load_data()

# model final untuk web/rekomendasi = K=2
hasil_all_k2, inertia_all_k2, dbi_all_k2, sil_all_k2, km_all_k2, centroid_all_k2 = proses_kmeans(data, 2)

# model pembanding = K=3
hasil_all_k3, inertia_all_k3, dbi_all_k3, sil_all_k3, km_all_k3, centroid_all_k3 = proses_kmeans(data, 3)

# data cluster utama untuk sistem rekomendasi menggunakan K=3
data_cluster = pd.merge(
    data,
    hasil_all_k3[["Produk", "cluster", "cluster_label", "qty", "frekuensi"]],
    on="Produk",
    how="left"
)

kategori_opsi = data_cluster["Kategori"].dropna().drop_duplicates().tolist()

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown('<div class="sidebar-box">', unsafe_allow_html=True)

    if os.path.exists(LOGO_PATH):
        st.markdown('<div class="sidebar-logo">', unsafe_allow_html=True)
        st.image(LOGO_PATH, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="sidebar-title">Dashboard<br><span>Menu</span></div>
        <div class="sidebar-subtitle">{APP_TITLE}</div>
        <div class="sidebar-divider"></div>
        <div class="sidebar-heading">Dashboard Menu</div>
        """,
        unsafe_allow_html=True
    )

    menu_options = [
        "Ringkasan Data",
        "Clustering",
        "Top 10 Produk",
        "Sistem Rekomendasi",
        "Input Transaksi"
    ]

    menu = st.radio(
        "",
        menu_options,
        index=0,
        key="menu_nav",
        label_visibility="collapsed"
    )

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# HALAMAN 1: INPUT TRANSAKSI
# =========================
if menu == "Input Transaksi":
    show_page_header(
        "Input Transaksi",
        "Masukkan transaksi penjualan ke file rekap."
    )

    if "cart" not in st.session_state:
        st.session_state.cart = []

    kode_preview = generate_kode(FILE_REKAP)

    with st.container(border=True):
        st.subheader(f"🧾 Kode Transaksi: {kode_preview}")

        produk_list = master["Produk"].dropna().drop_duplicates().tolist()

        col_a, col_b = st.columns([1, 2])
        with col_a:
            tanggal = st.date_input("Tanggal", datetime.today())
        with col_b:
            produk = st.selectbox("Pilih Produk", produk_list)

        info_produk = master[master["Produk"] == produk].drop_duplicates(subset=["Produk"])

        if not info_produk.empty:
            row = info_produk.iloc[0]
            kategori_produk = row["Kategori"]
            sub_kategori_produk = row["Sub Kategori"]
            target_produk = row["Target"]
            harga_jual = row["Harga Jual"]

            c1, c2, c3 = st.columns(3)
            c1.metric("Kategori", kategori_produk if pd.notna(kategori_produk) else "-")
            c2.metric("Sub Kategori", sub_kategori_produk if pd.notna(sub_kategori_produk) else "-")
            c3.metric("Target", target_produk if pd.notna(target_produk) else "-")

            if pd.notna(harga_jual):
                harga = int(float(harga_jual))
                st.success(f"Harga Jual: {format_rupiah(harga)}")
            else:
                harga = 0
                st.warning("Harga Jual belum tersedia.")
        else:
            harga = 0
            st.warning("Produk tidak ditemukan di master data.")

        jumlah = st.number_input("Jumlah", min_value=1, value=1)
        total = harga * jumlah
        st.info(f"Total Belanja: {format_rupiah(total)}")

        if st.button("➕ Tambah ke Keranjang"):
            st.session_state.cart.append({
                "Produk": produk,
                "Harga": harga,
                "Jumlah": jumlah,
                "Total": total
            })
            st.success("Produk berhasil ditambahkan ke keranjang.")

    st.divider()

    st.subheader("🧺 Keranjang")
    if st.session_state.cart:
        for i, item in enumerate(st.session_state.cart):
            with st.container(border=True):
                c1, c2, c3, c4, c5 = st.columns([4, 2, 2, 2, 1])

                c1.write(f"**{item['Produk']}**")
                c2.write(format_rupiah(item["Harga"]))
                c3.write(item["Jumlah"])
                c4.write(format_rupiah(item["Total"]))

                if c5.button("❌", key=f"hapus_{i}"):
                    st.session_state.cart.pop(i)
                    st.rerun()

        total_semua = sum(item["Total"] for item in st.session_state.cart)
        st.success(f"💰 Total Semua: {format_rupiah(total_semua)}")
    else:
        st.info("Keranjang masih kosong.")

    st.divider()

    if st.button("💾 Simpan Transaksi"):
        if not st.session_state.cart:
            st.warning("Keranjang kosong!")
        else:
            data_list = []
            for item in st.session_state.cart:
                data_list.append({
                    "Kode": kode_preview,
                    "Tanggal": tanggal,
                    "Produk": item["Produk"],
                    "Harga": item["Harga"],
                    "Jumlah": item["Jumlah"],
                    "Total": item["Total"]
                })

            df_save = pd.DataFrame(data_list)

            if os.path.exists(FILE_REKAP):
                df_existing = pd.read_excel(FILE_REKAP)
                df_final = pd.concat([df_existing, df_save], ignore_index=True)
                df_final.to_excel(FILE_REKAP, index=False)
            else:
                df_save.to_excel(FILE_REKAP, index=False)

            st.cache_data.clear()
            st.success(f"✅ Transaksi {kode_preview} berhasil disimpan!")
            st.session_state.cart = []
            st.rerun()

# =========================
# HALAMAN 2: RINGKASAN
# =========================
elif menu == "Ringkasan Data":
    show_page_header(
        "Sistem Toko Pertanian dan Pakan Ternak Mekarsari",
        "Ringkasan data transaksi dan hasil clustering."
    )

    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Transaksi", transaksi["Kode"].nunique())
        c2.metric("Produk Unik", data_cluster["Produk"].nunique())
        c3.metric("Kategori Unik", data_cluster["Kategori"].nunique(dropna=True))
        c4.metric("Sub Kategori Unik", data_cluster["Sub_Kategori"].nunique(dropna=True))

    st.divider()

    # =========================================================
    # JUMLAH DATA PER KATEGORI - TABEL + VISUAL
    # =========================================================
    st.subheader("Jumlah Data per Kategori")

    tabel_kategori = (
        data_cluster.groupby("Kategori", as_index=False)
        .agg(Jumlah_Data=("Produk", "size"))
        .sort_values("Jumlah_Data", ascending=False)
    )

    total_data = tabel_kategori["Jumlah_Data"].sum()
    tabel_kategori["Persentase"] = (
        tabel_kategori["Jumlah_Data"] / total_data * 100
    ).round(2)

    st.caption(
    f"Kategori dengan jumlah data terbanyak adalah **{tabel_kategori.iloc[0]['Kategori']}**."
)

    st.markdown("#### Tabel Jumlah Data per Kategori")
    st.dataframe(
        tabel_kategori.rename(columns={
            "Kategori": "Kategori",
            "Jumlah_Data": "Jumlah Data",
            "Persentase": "Persentase (%)"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("#### Visualisasi")
    fig_kat, ax_kat = plt.subplots(figsize=(6, 2.8))
    bars = ax_kat.barh(tabel_kategori["Kategori"], tabel_kategori["Jumlah_Data"])
    ax_kat.invert_yaxis()
    ax_kat.set_title("Jumlah Data per Kategori", fontsize=13, fontweight="bold")
    ax_kat.set_xlabel("Jumlah Data")
    ax_kat.set_ylabel("Kategori")
    ax_kat.grid(axis="x", linestyle="--", alpha=0.25)

    st.pyplot(fig_kat, use_container_width=True)

    st.divider()

    # =========================================================
    # JUMLAH DATA PER SUB KATEGORI
    # =========================================================
    st.subheader("Jumlah Data per Sub Kategori")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Pertanian")
        tabel_pertanian = (
            data_cluster[data_cluster["Kategori"] == "Pertanian"]
            .dropna(subset=["Sub_Kategori"])
            .groupby("Sub_Kategori")
            .size()
            .reset_index(name="Jumlah Data")
            .sort_values("Jumlah Data", ascending=False)
        )
        st.dataframe(
            tabel_pertanian,
            use_container_width=True,
            hide_index=True,
            height=320
        )

    with col2:
        st.markdown("#### Peternakan")
        tabel_peternakan = (
            data_cluster[data_cluster["Kategori"] == "Peternakan"]
            .dropna(subset=["Sub_Kategori"])
            .groupby("Sub_Kategori")
            .size()
            .reset_index(name="Jumlah Data")
            .sort_values("Jumlah Data", ascending=False)
        )
        st.dataframe(
            tabel_peternakan,
            use_container_width=True,
            hide_index=True,
            height=320
        )

    with st.expander("Lihat Data Setelah Merge"):
        tampil_merge = data_cluster[
            [
                "Kode",
                "Tanggal",
                "Produk",
                "Harga",
                "Jumlah",
                "Total",
                "Kategori",
                "Sub_Kategori",
                "Target",
                "Harga_Jual",
                "qty",
                "frekuensi"
            ]
        ]

    st.dataframe(
        tampil_merge.head(20),
        use_container_width=True,
        hide_index=True
    )

# =========================
# HALAMAN 3: CLUSTERING
# =========================
elif menu == "Clustering":
    show_page_header(
        "Hasil Clustering",
        "Analisis produk berdasarkan quantity dan frekuensi."
    )

    pilih_dataset = st.selectbox("Pilih Data", ["Gabungan", "Pertanian", "Peternakan"])
    pilih_k = st.selectbox("Pilih K", [2, 3])

    if pilih_dataset == "Gabungan":
        df_source = data.copy()
        title = "Gabungan"
    else:
        df_source = data[data["Kategori"] == pilih_dataset].copy()
        title = pilih_dataset

    # hitung K sesuai pilihan user
    df_plot, inertia, dbi, sil, km, centroid = proses_kmeans(df_source, pilih_k)
    centroid_df = pd.DataFrame(centroid, columns=["qty", "frekuensi"])

    c1, c2 = st.columns(2)
    c1.metric("Davies-Bouldin Index", f"{dbi:.4f}")
    c2.metric("Silhouette Score", f"{sil:.4f}")

    st.divider()

    # elbow method
    st.subheader(f"Elbow Method - {title}")
    fig1, ax1 = plt.subplots(figsize=(6.5, 3.5))
    ax1.plot(range(1, 6), inertia[:5], marker="o")
    apply_plot_style(ax1, f"Elbow - {title}", "Jumlah Cluster", "Inertia")
    st.pyplot(fig1, use_container_width=True)

    # scatter
    st.subheader(f"Scatter Plot - {title} (K={pilih_k})")
    fig2, ax2 = plt.subplots(figsize=(6.5, 4))
    ax2.scatter(
        df_plot["qty"],
        df_plot["frekuensi"],
        c=df_plot["cluster"],
        cmap="viridis",
        s=50,
        alpha=0.85
    )
    apply_plot_style(ax2, f"Clustering Produk - {title}", "Total Qty", "Frekuensi")
    st.pyplot(fig2, use_container_width=True)

    # ringkasan cluster
    st.subheader("Ringkasan Cluster")
    ringkasan = (
        df_plot.groupby("cluster_label", as_index=False)
        .agg(
            Jumlah_Produk=("Produk", "nunique"),
            Rata_Rata_Qty=("qty", "mean"),
            Rata_Rata_Frekuensi=("frekuensi", "mean")
        )
        .sort_values("Rata_Rata_Qty", ascending=False)
    )

    ringkasan["Persentase"] = (
        ringkasan["Jumlah_Produk"] / ringkasan["Jumlah_Produk"].sum() * 100
    ).round(2)

    ringkasan = ringkasan.rename(columns={"cluster_label": "Cluster"})
    ringkasan["Rata_Rata_Qty"] = ringkasan["Rata_Rata_Qty"].round(2)
    ringkasan["Rata_Rata_Frekuensi"] = ringkasan["Rata_Rata_Frekuensi"].round(2)

    st.dataframe(ringkasan, use_container_width=True, hide_index=True, height=220)

    # visual jumlah cluster kecil
    st.subheader("Jumlah Produk per Cluster")
    fig3, ax3 = plt.subplots(figsize=(6.5, 3.5))
    df_plot["cluster_label"].value_counts().sort_values(ascending=True).plot(kind="barh", ax=ax3)
    apply_plot_style(ax3, f"Jumlah Produk per Cluster - {title}", "Jumlah Produk", "Cluster")
    st.pyplot(fig3, use_container_width=True)

    # detail produk per cluster
    st.subheader("Detail Produk per Cluster")
    detail = df_plot[["Produk", "qty", "frekuensi", "cluster_label"]].copy()
    detail = detail.sort_values(["cluster_label", "qty"], ascending=[True, False])
    detail = detail.rename(columns={
        "qty": "Total Qty",
        "frekuensi": "Frekuensi",
        "cluster_label": "Cluster"
    })
    st.dataframe(detail, use_container_width=True, hide_index=True, height=320)

    # centroid
    st.subheader("Centroid (Original Scale)")
    centroid_df = centroid_df.round(2)
    st.dataframe(centroid_df, use_container_width=True, hide_index=True)

    # perbandingan K=2 vs K=3 untuk subset yang dipilih
    st.subheader("Perbandingan K=2 dan K=3")

    hasil_k2, inertia_k2, dbi_k2, sil_k2, km_k2, centroid_k2 = proses_kmeans(df_source, 2)
    hasil_k3, inertia_k3, dbi_k3, sil_k3, km_k3, centroid_k3 = proses_kmeans(df_source, 3)

    df_band = pd.DataFrame([
        {"K": 2, "Davies-Bouldin Index": dbi_k2, "Silhouette Score": sil_k2},
        {"K": 3, "Davies-Bouldin Index": dbi_k3, "Silhouette Score": sil_k3},
    ])

    st.dataframe(df_band, use_container_width=True, hide_index=True)

# =========================
# HALAMAN 4: TOP 10 PRODUK
# =========================
elif menu == "Top 10 Produk":
    show_page_header(
        "Top 10 Produk Terlaris",
        "Menampilkan produk dengan total penjualan tertinggi."
    )

    pilih_kategori = st.selectbox("Pilih Kategori", ["Gabungan"] + kategori_opsi)

    if pilih_kategori == "Gabungan":
        df_top = data_cluster.copy()
    else:
        df_top = data_cluster[data_cluster["Kategori"] == pilih_kategori].copy()

    top_produk = (
        df_top.groupby("Produk", as_index=False)["Jumlah"]
        .sum()
        .sort_values("Jumlah", ascending=False)
        .head(10)
    )

    st.subheader("Tabel Top 10 Produk")
    st.dataframe(top_produk, use_container_width=True, hide_index=True)

    st.subheader("Grafik Top 10 Produk Terlaris")
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    ax4.barh(top_produk["Produk"], top_produk["Jumlah"])
    ax4.invert_yaxis()
    apply_plot_style(ax4, "Top 10 Produk Terlaris", "Total Jumlah", "Produk")
    st.pyplot(fig4, use_container_width=True)

# =========================
# HALAMAN 5: SISTEM REKOMENDASI
# =========================
elif menu == "Sistem Rekomendasi":
    show_page_header(
        "Sistem Rekomendasi Produk",
        "Rekomendasi produk untuk pelanggan berdasarkan kategori, sub kategori, dan target."
    )

    if not kategori_opsi:
        st.warning("Data kategori tidak tersedia.")
    else:
        kategori = st.selectbox("Pilih Kategori", kategori_opsi)

        sub_opsi = (
            data_cluster[data_cluster["Kategori"] == kategori]["Sub_Kategori"]
            .dropna()
            .drop_duplicates()
            .tolist()
        )

        if not sub_opsi:
            st.warning("Sub kategori tidak tersedia untuk kategori ini.")
        else:
            sub = st.selectbox("Pilih Sub Kategori", sub_opsi)

            target_opsi = get_target_options(data_cluster, kategori, sub)
            target = st.selectbox("Pilih Target", target_opsi)

            hasil = data_cluster[
                (data_cluster["Kategori"] == kategori) &
                (data_cluster["Sub_Kategori"] == sub)
            ].copy()

            hasil["Target_List"] = hasil["Target"].apply(split_targets)

            if target == "umum":
                hasil_filter = hasil.copy()
            else:
                hasil_filter = hasil[
                    hasil["Target_List"].apply(lambda x: target in x or "umum" in x)
                ].copy()

            if hasil_filter.empty:
                st.warning("Tidak ada rekomendasi untuk pilihan ini.")
            else:
                rekom = hasil_filter[[
                    "Produk", "Harga_Jual", "Target",
                    "cluster", "cluster_label", "qty", "frekuensi"
                ]].copy()

                rekom = rekom.drop_duplicates(subset=["Produk"])

                # urutan cluster: Laris dulu, lalu Kurang Laris
                prioritas_map = {
                    "Laris": 1,
                    "Kurang Laris": 2
                }

                rekom["Prioritas"] = rekom["cluster_label"].map(prioritas_map).fillna(99).astype(int)

                rekom = rekom.sort_values(
                    by=["Prioritas", "qty", "frekuensi"],
                    ascending=[True, False, False]
                )

                st.subheader("Rekomendasi Produk")
                st.write(f"**Kategori:** {kategori}")
                st.write(f"**Sub Kategori:** {sub}")
                st.write(f"**Target:** {target}")
                st.info("Urutan rekomendasi ditampilkan dari cluster Laris ke Kurang Laris.")

                tampil = rekom[["Produk", "qty", "frekuensi", "Harga_Jual", "Target"]].copy()
                tampil = tampil.rename(columns={
                    "Produk": "Nama Barang",
                    "qty": "Total Qty",
                    "frekuensi": "Frekuensi",
                    "Harga_Jual": "Harga Jual"
                })

                tampil["Harga Jual"] = tampil["Harga Jual"].apply(format_rupiah)

                # tinggi tabel otomatis mengikuti jumlah data
                jumlah_baris = len(tampil)
                tinggi_tabel = min(80 + (jumlah_baris * 38), 260)
                
                st.dataframe(
                    tampil,
                    use_container_width=True,
                    hide_index=True,
                    height=tinggi_tabel
                )

                st.download_button(
                    "Download Hasil Rekomendasi",
                    data=tampil.to_csv(index=False).encode("utf-8"),
                    file_name=f"rekomendasi_{kategori}_{sub}_{target}.csv",
                    mime="text/csv"
                )
