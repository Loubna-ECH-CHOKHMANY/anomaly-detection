import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import median_abs_deviation

# ════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION DE LA PAGE
# ════════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Détection d'Anomalies",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════════════════════════════
# CSS GLOBAL — thème sombre, titres blancs
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    h1, h2, h3, h4, h5, h6,
    .main-title, .section-header {
        color: #ffffff !important;
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #a0aec0;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.35rem;
        font-weight: 600;
        color: #ffffff !important;
        border-left: 4px solid #667eea;
        padding-left: 0.75rem;
        margin: 1.8rem 0 1rem 0;
    }
    .info-box {
        background: #1e2a3a;
        border-left: 4px solid #3498db;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        color: #d0e4f7;
    }
    .warning-box {
        background: #2a1f0e;
        border-left: 4px solid #ff9800;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        color: #ffe0a0;
    }
    .success-box {
        background: #0e2a14;
        border-left: 4px solid #4caf50;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        color: #a5d6a7;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }
    .stDownloadButton > button {
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# SESSION STATE — mémorisation des résultats entre les rechargements
# ════════════════════════════════════════════════════════════════════════════════
if "results" not in st.session_state:
    st.session_state.results = None

# ════════════════════════════════════════════════════════════════════════════════
# MODULE 1 — DÉTECTION DES COLONNES TEMPORELLES
# ════════════════════════════════════════════════════════════════════════════════
def detect_time_column(df: pd.DataFrame) -> str | None:
    """Cherche une colonne temporelle (timestamp, date, datetime, time) — silencieux."""
    for col in df.columns:
        if col.strip().lower() in ("timestamp", "date", "datetime", "time"):
            return col
    return None


def get_x_axis(df: pd.DataFrame, series: pd.Series, time_col: str | None):
    """Retourne (x_values, x_label) selon qu'une colonne temporelle existe ou non."""
    if time_col and time_col in df.columns:
        return df.loc[series.index, time_col], time_col
    return series.index, "Index"


# ════════════════════════════════════════════════════════════════════════════════
# MODULE 2 — MÉTHODES DE DÉTECTION
# ════════════════════════════════════════════════════════════════════════════════
def detect_zscore_robust(series: pd.Series, threshold: float):
    """Z-score robuste : (x - médiane) / MAD."""
    median = series.median()
    mad = median_abs_deviation(series, scale="normal")
    if mad == 0:
        z = pd.Series(np.zeros(len(series)), index=series.index)
        return z.abs() > threshold, z, median, mad
    z = (series - median) / mad
    return z.abs() > threshold, z, median, mad


def detect_manual(series: pd.Series, low: float, high: float):
    """Seuil manuel défini par l'utilisateur."""
    median = series.median()
    mad = median_abs_deviation(series, scale="normal")
    return (series < low) | (series > high), None, median, mad


def interpret_anomaly(value: float, center: float, spread: float) -> str:
    if spread > 0 and value > center + 2 * spread:
        return "⬆️ Anormalement élevée"
    elif spread > 0 and value < center - 2 * spread:
        return "⬇️ Anormalement basse"
    return "⚠️ Hors seuil"


# ════════════════════════════════════════════════════════════════════════════════
# MODULE 3 — NETTOYAGE DES DONNÉES
# ════════════════════════════════════════════════════════════════════════════════
def clean_replace_median(df: pd.DataFrame, anomalies_mask: pd.Series, col: str) -> pd.DataFrame:
    """Remplace les valeurs anomalies par la médiane de la colonne."""
    out = df.copy()
    mask = anomalies_mask.reindex(out.index, fill_value=False)
    out.loc[mask, col] = round(out[col].median(), 4)
    return out


def clean_drop_anomalies(df: pd.DataFrame, anomalies_mask: pd.Series) -> pd.DataFrame:
    """Supprime les lignes contenant des anomalies."""
    out = df.copy()
    mask = anomalies_mask.reindex(out.index, fill_value=False)
    return out[~mask].reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════════
# MODULE 4 — VISUALISATION
# ════════════════════════════════════════════════════════════════════════════════
def build_main_figure(
    series: pd.Series,
    anomalies: pd.Series,
    x_vals,
    x_label: str,
    column: str,
    center: float,
) -> go.Figure:
    """Graphique principal : courbe + points normaux bleus + anomalies rouges."""
    x_list      = list(x_vals)
    normal_mask = ~anomalies
    fig = go.Figure()

    # Ligne de fond — pas de tooltip
    fig.add_trace(go.Scatter(
        x=x_list, y=list(series.values),
        mode="lines", name="Données",
        line=dict(color="#3a86ff", width=1.5), opacity=0.45,
        hoverinfo="skip"
    ))

    # Points normaux — tooltip : date/heure + valeur
    fig.add_trace(go.Scatter(
        x=[x_list[i] for i in range(len(series)) if normal_mask.iloc[i]],
        y=list(series[normal_mask].values),
        mode="markers", name="Valeur normale",
        marker=dict(color="#3a86ff", size=6, symbol="circle"),
        hovertemplate=(
            f"<b>📅 {x_label}</b> : %{{x}}<br>"
            f"<b>📏 {column}</b> : %{{y:.4f}}"
            "<extra>Valeur normale</extra>"
        )
    ))

    # Points anomalies — tooltip : date/heure + valeur
    anom_x = [x_list[i] for i in range(len(series)) if anomalies.iloc[i]]
    anom_y = list(series[anomalies].values)
    if anom_x:
        fig.add_trace(go.Scatter(
            x=anom_x, y=anom_y,
            mode="markers", name="Anomalie",
            marker=dict(color="#ef233c", size=13, symbol="x",
                        line=dict(width=2.5, color="#ef233c")),
            hovertemplate=(
                f"<b>📅 {x_label}</b> : %{{x}}<br>"
                f"<b>📏 {column}</b> : %{{y:.4f}}"
                "<extra>⚠️ Anomalie</extra>"
            )
        ))

    # Ligne médiane de référence
    fig.add_hline(
        y=center, line_dash="dash", line_color="#94a3b8",
        annotation_text=f"Médiane : {center:.2f}",
        annotation_position="top right"
    )

    fig.update_layout(
        title=dict(
            text=f"📈 Série temporelle — <b>{column}</b>",
            font=dict(size=16, color="#ffffff")
        ),
        xaxis_title=x_label, yaxis_title=column,
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        font=dict(color="#cbd5e1"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(color="#ffffff")
        ),
        margin=dict(t=80, b=40, l=40, r=40),
        hovermode="closest"
    )
    fig.update_xaxes(showgrid=True, gridcolor="#1e293b", color="#94a3b8")
    fig.update_yaxes(showgrid=True, gridcolor="#1e293b", color="#94a3b8")
    return fig


# ════════════════════════════════════════════════════════════════════════════════
# EN-TÊTE
# ════════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-title">🔍 Application de Détection d\'Anomalies</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyse statistique — Z-score Robuste (MAD) & seuil manuel</div>',
            unsafe_allow_html=True)
st.divider()


# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR — épurée : import + méthode + bouton analyser uniquement
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Paramètres d'analyse")

    st.markdown("### 📂 Importation des données")
    uploaded_file = st.file_uploader("Importer un fichier CSV", type=["csv"])

    st.markdown("---")

    st.markdown("### 🧠 Méthode de détection")
    method = st.radio(
        "Choisir la méthode :",
        ["🛡️ Z-score Robuste (MAD)", "✏️ Seuil Manuel"],
        index=0,
        help="Le Z-score robuste utilise la médiane et le MAD, résistants aux outliers extrêmes."
    )

    threshold   = 2.5
    manual_low  = 0.0
    manual_high = 100.0

    if method == "🛡️ Z-score Robuste (MAD)":
        threshold = st.slider(
            "Seuil Z-score", min_value=1.5, max_value=4.0,
            value=2.5, step=0.05, help="|z| > seuil → anomalie"
        )
        st.caption(f"Seuil actuel : {threshold} — valeurs à plus de {threshold}×MAD de la médiane")
    else:
        manual_low  = st.number_input("Seuil bas  (minimum acceptable)", value=0.0,   step=1.0)
        manual_high = st.number_input("Seuil haut (maximum acceptable)", value=100.0, step=1.0)

    st.markdown("---")
    analyze_btn = st.button("🔎 Analyser", use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# ZONE PRINCIPALE
# ════════════════════════════════════════════════════════════════════════════════
if uploaded_file is None:
    _, c, _ = st.columns([1, 2, 1])
    with c:
        st.markdown("""
        <div class="info-box">
        <b>👈 Comment commencer ?</b><br><br>
        1. Importez un fichier CSV depuis la barre latérale<br>
        2. Sélectionnez la colonne à analyser<br>
        3. Choisissez la méthode de détection<br>
        4. Cliquez sur <b>Analyser</b>
        </div>
        <div class="info-box">
        <b>📋 Format attendu</b><br><br>
        Le fichier doit contenir au moins une colonne numérique.
        </div>
        """, unsafe_allow_html=True)

else:
    # ── Lecture ──────────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Erreur de lecture : {e}")
        st.stop()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("❌ Aucune colonne numérique détectée dans le fichier.")
        st.stop()

    # Détection silencieuse de la colonne temporelle
    time_col = detect_time_column(df)

    # ── Sélection de colonne + aperçu ────────────────────────────────────────────
    col_sel, col_prev = st.columns([1, 2])
    with col_sel:
        st.markdown('<div class="section-header">📋 Sélection de la colonne</div>',
                    unsafe_allow_html=True)
        selected_col = st.selectbox("Colonne à analyser", numeric_cols)
    with col_prev:
        st.markdown('<div class="section-header">👁️ Aperçu des données</div>',
                    unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True, height=220)

    series = df[selected_col].dropna()

    # ════════════════════════════════════════════════════════════════════════════
    # ANALYSE
    # ════════════════════════════════════════════════════════════════════════════
    if analyze_btn:
        # ── Détection ────────────────────────────────────────────────────────────
        if method == "🛡️ Z-score Robuste (MAD)":
            anomalies_mask, z_scores, center, spread = detect_zscore_robust(series, threshold)
            method_label = "Z-score Robuste (Médiane / MAD)"
        else:
            anomalies_mask, z_scores, center, spread = detect_manual(series, manual_low, manual_high)
            method_label = "Seuil Manuel"

        # ── Sauvegarder dans session_state ───────────────────────────────────────
        st.session_state.results = {
            "df":             df,
            "series":         series,
            "selected_col":   selected_col,
            "anomalies_mask": anomalies_mask,
            "z_scores":       z_scores,
            "center":         center,
            "spread":         spread,
            "method_label":   method_label,
            "time_col":       time_col,
        }

    # ── Afficher les résultats (depuis session_state) ─────────────────────────
    if st.session_state.results is not None:
        r             = st.session_state.results
        df            = r["df"]
        series        = r["series"]
        selected_col  = r["selected_col"]
        anomalies_mask= r["anomalies_mask"]
        z_scores      = r["z_scores"]
        center        = r["center"]
        spread        = r["spread"]
        method_label  = r["method_label"]
        time_col      = r["time_col"]

        n_anomalies = int(anomalies_mask.sum())
        n_total     = len(series)

        # Axe X (timestamp ou index numérique)
        x_vals, x_label = get_x_axis(df, series, time_col)

        st.divider()

        # ── § 1 — RÉSUMÉ STATISTIQUE ─────────────────────────────────────────────
        st.markdown('<div class="section-header">📊 Résumé statistique</div>',
                    unsafe_allow_html=True)

        metrics = [
            ("Total valeurs", str(n_total),                              "#667eea"),
            ("Médiane",       f"{series.median():.3f}",                  "#764ba2"),
            ("Écart-type",    f"{series.std():.3f}",                     "#3a86ff"),
            ("Min / Max",     f"{series.min():.2f} / {series.max():.2f}","#0ea5e9"),
            ("Anomalies",     str(n_anomalies),                          "#ef233c"),
        ]
        stat_cols = st.columns(len(metrics))
        for col_obj, (label, value, color) in zip(stat_cols, metrics):
            with col_obj:
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,{color}bb,{color});
                    border-radius:12px;padding:1.1rem;color:white;text-align:center;
                    box-shadow:0 4px 15px {color}44;">
                    <div style="font-size:1.7rem;font-weight:700">{value}</div>
                    <div style="font-size:0.78rem;opacity:0.9;margin-top:0.3rem">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── § 2 — GRAPHIQUE ──────────────────────────────────────────────────────
        st.markdown('<div class="section-header">📈 Visualisation graphique</div>',
                    unsafe_allow_html=True)

        fig = build_main_figure(series, anomalies_mask, x_vals, x_label, selected_col, center)
        st.plotly_chart(fig, use_container_width=True)

        # ── § 3 — TABLEAU DES ANOMALIES ──────────────────────────────────────────
        st.markdown('<div class="section-header">⚠️ Anomalies détectées</div>',
                    unsafe_allow_html=True)

        if n_anomalies == 0:
            st.markdown("""
            <div class="success-box">
            ✅ <b>Aucune anomalie détectée</b> avec les paramètres actuels.<br>
            Essayez de réduire le seuil ou de changer de méthode.
            </div>
            """, unsafe_allow_html=True)
        else:
            anom_series = series[anomalies_mask]

            if time_col and time_col in df.columns:
                index_vals = df.loc[anom_series.index, time_col].values
                index_name = time_col
            else:
                index_vals = anom_series.index.tolist()
                index_name = "Index"

            anom_df = pd.DataFrame({
                index_name:            index_vals,
                "Valeur":              anom_series.values,
                "Écart à la médiane":  (anom_series.values - center),
            })
            if z_scores is not None:
                anom_df["Z-score (MAD)"] = z_scores[anomalies_mask].values.round(4)
            anom_df["Interprétation"] = [
                interpret_anomaly(v, center, spread) for v in anom_series.values
            ]

            st.dataframe(
                anom_df.style.format({
                    "Valeur":             "{:.4f}",
                    "Écart à la médiane": "{:+.4f}",
                }),
                use_container_width=True,
                hide_index=True
            )

            pct   = (n_anomalies / n_total) * 100
            level = "🟢 faible" if pct < 5 else ("🟡 modéré" if pct < 15 else "🔴 élevé")
            st.markdown(f"""
            <div class="warning-box">
            ⚠️ <b>{n_anomalies} anomalie(s) détectée(s)</b> sur {n_total} valeurs
            ({pct:.1f}%) — taux {level}<br>
            <small>Méthode : {method_label} | Médiane de référence : {center:.4f}</small>
            </div>
            """, unsafe_allow_html=True)

        # ── § 4 — NETTOYAGE & EXPORT (3 boutons) ─────────────────────────────────
        st.markdown('<div class="section-header">💾 Nettoyage & Export</div>',
                    unsafe_allow_html=True)

        # Préparer les 3 datasets
        # 1) Uniquement les anomalies (avec statut et z-score)
        export_full = df.copy()
        export_full["statut"] = anomalies_mask.reindex(df.index, fill_value=False) \
                                              .map({True: "anomalie", False: "normal"})
        if z_scores is not None:
            export_full["z_score_mad"] = z_scores.reindex(df.index).round(4)
        df_anomalies_only = export_full[export_full["statut"] == "anomalie"].copy()

        # 2) Dataset sans les lignes anomalies
        df_supprime = clean_drop_anomalies(df, anomalies_mask)

        # 3) Dataset avec anomalies remplacées par la médiane
        df_mediane  = clean_replace_median(df, anomalies_mask, selected_col)

        dl1, dl2, dl3 = st.columns(3)

        with dl1:
            st.download_button(
                label="⬇️ Base des anomalies détectées",
                data=df_anomalies_only.to_csv(index=False),
                file_name="anomalies_detectees.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.markdown(f"""
            <div class="info-box" style="font-size:0.82rem;margin-top:0.5rem">
            📋 <b>{n_anomalies} ligne(s)</b> anormale(s) uniquement,
            avec colonnes <b>statut</b> et <b>Z-score MAD</b>.
            </div>
            """, unsafe_allow_html=True)

        with dl2:
            st.download_button(
                label="🗑️ Base nettoyée (anomalies supprimées)",
                data=df_supprime.to_csv(index=False),
                file_name="donnees_supprimees.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.markdown(f"""
            <div class="info-box" style="font-size:0.82rem;margin-top:0.5rem">
            🗑️ Dataset réduit à <b>{n_total - n_anomalies} valeur(s)</b>
            ({n_anomalies} ligne(s) anormale(s) retirée(s)).
            </div>
            """, unsafe_allow_html=True)

        with dl3:
            st.download_button(
                label="🔄 Base nettoyée (remplacées par la médiane)",
                data=df_mediane.to_csv(index=False),
                file_name="donnees_mediane.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.markdown(f"""
            <div class="info-box" style="font-size:0.82rem;margin-top:0.5rem">
            📐 Anomalies remplacées par la médiane
            <b>{series.median():.4f}</b> ({n_anomalies} valeur(s) modifiée(s)).
            </div>
            """, unsafe_allow_html=True)
