import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="COVID-19 Dashboard Analytics",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# ESTILOS CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Fondo general */
    .stApp { background-color: #0d1117; color: #e6edf3; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
        border-right: 1px solid #30363d;
    }

    /* Tarjetas de métricas */
    .metric-card {
        background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.4);
        margin-bottom: 12px;
    }
    .metric-card .label {
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: #8b949e;
        margin-bottom: 6px;
    }
    .metric-card .value {
        font-size: 28px;
        font-weight: 700;
        color: #e6edf3;
    }
    .metric-card .delta {
        font-size: 12px;
        margin-top: 4px;
    }
    .delta-pos { color: #3fb950; }
    .delta-neg { color: #f85149; }
    .delta-neu { color: #58a6ff; }

    /* Encabezados de sección */
    .section-header {
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #58a6ff;
        border-bottom: 1px solid #21262d;
        padding-bottom: 8px;
        margin: 28px 0 16px 0;
    }

    /* Tabla de estadísticas */
    .stat-table { width: 100%; border-collapse: collapse; font-size: 13px; }
    .stat-table th {
        background: #21262d;
        color: #8b949e;
        padding: 8px 12px;
        text-align: left;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .stat-table td {
        padding: 7px 12px;
        border-bottom: 1px solid #21262d;
        color: #c9d1d9;
    }
    .stat-table tr:hover td { background: #161b22; }

    /* Pills de categorías */
    .pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        margin: 2px;
    }
    .pill-blue  { background: #1f4068; color: #79c0ff; }
    .pill-green { background: #0f3d1e; color: #3fb950; }
    .pill-red   { background: #3d0f0f; color: #f85149; }
    .pill-orange{ background: #3d280f; color: #f0883e; }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #1a237e 0%, #0d1117 60%, #1b5e20 100%);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 28px;
    }
    .hero-title {
        font-size: 32px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #e6edf3;
        margin: 0 0 6px 0;
    }
    .hero-sub {
        font-size: 14px;
        color: #8b949e;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# GENERACIÓN DE DATOS SINTÉTICOS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Generando datos sintéticos…")
def generate_covid_data(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ── 1. Fecha de diagnóstico (datetime) ──────────────────────────────
    start = np.datetime64("2020-01-01")
    days  = (np.datetime64("2023-12-31") - start).astype(int)
    fechas = start + rng.integers(0, days, size=n).astype("timedelta64[D]")

    # ── 2. Edad (int) ───────────────────────────────────────────────────
    edades = rng.integers(1, 99, size=n)

    # ── 3. Género (categorical) ─────────────────────────────────────────
    genero = rng.choice(["Masculino", "Femenino", "Otro"],
                        p=[0.49, 0.49, 0.02], size=n)

    # ── 4. Departamento (categorical) ───────────────────────────────────
    departamentos = [
        "Antioquia", "Bogotá", "Valle del Cauca", "Atlántico",
        "Santander", "Cundinamarca", "Bolívar", "Nariño",
        "Córdoba", "Tolima",
    ]
    dpto_probs = [0.18, 0.22, 0.14, 0.10, 0.07,
                  0.09, 0.06, 0.05, 0.05, 0.04]
    departamento = rng.choice(departamentos, p=dpto_probs, size=n)

    # ── 5. Variante (categorical) ────────────────────────────────────────
    variante = rng.choice(
        ["Original", "Alpha", "Delta", "Ómicron", "Otra"],
        p=[0.20, 0.15, 0.25, 0.35, 0.05], size=n,
    )

    # ── 6. Días hospitalizado (float, con NaN si no hospitalizó) ────────
    hospitalizacion_prob = rng.random(n)
    dias_hosp = np.where(
        hospitalizacion_prob < 0.30,
        rng.exponential(scale=6, size=n).round(1),
        np.nan,
    )

    # ── 7. Resultado PCR (bool / categorical) ───────────────────────────
    resultado_pcr = rng.choice(["Positivo", "Negativo"], p=[0.68, 0.32], size=n)

    # ── 8. Desenlace (categorical, correlacionado con edad) ─────────────
    prob_fallecido = np.clip(0.005 + (edades / 100) * 0.12, 0, 0.20)
    prob_critico   = np.clip(0.02  + (edades / 100) * 0.08, 0, 0.15)
    prob_grave     = np.full(n, 0.10)
    prob_leve      = np.clip(
        1 - prob_fallecido - prob_critico - prob_grave, 0.01, 1
    )
    total = prob_fallecido + prob_critico + prob_grave + prob_leve
    prob_matrix = np.column_stack([
        prob_leve / total,
        prob_grave / total,
        prob_critico / total,
        prob_fallecido / total,
    ])
    desenlace_idx = np.array([
        rng.choice(4, p=prob_matrix[i]) for i in range(n)
    ])
    desenlace = np.array(["Leve", "Grave", "Crítico", "Fallecido"])[desenlace_idx]

    df = pd.DataFrame({
        "fecha_diagnostico": fechas.astype("datetime64[ms]"),
        "edad":              edades,
        "genero":            genero,
        "departamento":      departamento,
        "variante":          variante,
        "dias_hospitalizado": dias_hosp,
        "resultado_pcr":     resultado_pcr,
        "desenlace":         desenlace,
    })

    # Columnas de tiempo derivadas (útiles para filtros)
    df["año"]  = df["fecha_diagnostico"].dt.year
    df["mes"]  = df["fecha_diagnostico"].dt.month
    df["mes_nombre"] = df["fecha_diagnostico"].dt.strftime("%b %Y")

    return df


# ─────────────────────────────────────────────
# SIDEBAR — FILTROS
# ─────────────────────────────────────────────
def build_sidebar(df: pd.DataFrame):
    st.sidebar.markdown("## 🎛️ Filtros")

    años = sorted(df["año"].unique())
    año_sel = st.sidebar.multiselect(
        "Año de diagnóstico", años, default=años, key="año"
    )

    dptos = sorted(df["departamento"].unique())
    dpto_sel = st.sidebar.multiselect(
        "Departamento", dptos, default=dptos, key="dpto"
    )

    generos = sorted(df["genero"].unique())
    gen_sel = st.sidebar.multiselect(
        "Género", generos, default=generos, key="gen"
    )

    variantes = sorted(df["variante"].unique())
    var_sel = st.sidebar.multiselect(
        "Variante", variantes, default=variantes, key="var"
    )

    edad_min, edad_max = int(df["edad"].min()), int(df["edad"].max())
    edad_rng = st.sidebar.slider(
        "Rango de edad", edad_min, edad_max, (edad_min, edad_max)
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<small style='color:#8b949e'>Datos 100 % sintéticos generados con NumPy.<br>"
        "No representan casos reales.</small>",
        unsafe_allow_html=True,
    )

    mask = (
        df["año"].isin(año_sel)
        & df["departamento"].isin(dpto_sel)
        & df["genero"].isin(gen_sel)
        & df["variante"].isin(var_sel)
        & df["edad"].between(*edad_rng)
    )
    return df[mask].copy()


# ─────────────────────────────────────────────
# COLORES COMPARTIDOS
# ─────────────────────────────────────────────
PALETTE = {
    "Leve":      "#3fb950",
    "Grave":     "#f0883e",
    "Crítico":   "#d29922",
    "Fallecido": "#f85149",
    "Positivo":  "#58a6ff",
    "Negativo":  "#8b949e",
    "Masculino": "#79c0ff",
    "Femenino":  "#f778ba",
    "Otro":      "#d2a8ff",
    "Original":  "#3fb950",
    "Alpha":     "#58a6ff",
    "Delta":     "#f0883e",
    "Ómicron":   "#d29922",
    "Otra":      "#8b949e",
}
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c9d1d9", size=12),
    margin=dict(t=40, b=40, l=40, r=20),
    legend=dict(
        bgcolor="rgba(22,27,34,0.8)",
        bordercolor="#30363d",
        borderwidth=1,
    ),
)


# ─────────────────────────────────────────────
# HELPERS DE FORMATO
# ─────────────────────────────────────────────
def fmt(n): return f"{n:,.0f}"
def pct(a, b): return f"{a/b*100:.1f}%" if b else "—"


# ─────────────────────────────────────────────
# SECCIÓN 1 — MÉTRICAS GLOBALES
# ─────────────────────────────────────────────
def render_kpis(df: pd.DataFrame, df_full: pd.DataFrame):
    st.markdown('<div class="section-header">📊 Indicadores Globales</div>', unsafe_allow_html=True)

    total      = len(df)
    positivos  = (df["resultado_pcr"] == "Positivo").sum()
    fallecidos = (df["desenlace"] == "Fallecido").sum()
    hospitalizados = df["dias_hospitalizado"].notna().sum()
    edad_media = df["edad"].mean()
    dias_media = df["dias_hospitalizado"].mean()
    tasa_mort  = fallecidos / total * 100 if total else 0
    tasa_hosp  = hospitalizados / total * 100 if total else 0

    cols = st.columns(4)
    kpis = [
        ("Total Registros",      fmt(total),            f"{pct(total, len(df_full))} del universo", "neu"),
        ("Casos Positivos PCR",  fmt(positivos),        f"{pct(positivos, total)} positividad",     "neg"),
        ("Fallecidos",           fmt(fallecidos),       f"{tasa_mort:.2f}% letalidad",              "neg"),
        ("Hospitalizados",       fmt(hospitalizados),   f"{tasa_hosp:.1f}% del total",              "neu"),
    ]
    for col, (label, value, delta, typ) in zip(cols, kpis):
        with col:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="label">{label}</div>
                    <div class="value">{value}</div>
                    <div class="delta delta-{typ}">{delta}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    cols2 = st.columns(4)
    kpis2 = [
        ("Edad Promedio",        f"{edad_media:.1f} años",  "de los pacientes",         "neu"),
        ("Días Hosp. Promedio",  f"{dias_media:.1f} días" if not np.isnan(dias_media) else "—",
                                 "entre hospitalizados",      "neu"),
        ("Casos Críticos",       fmt((df["desenlace"]=="Crítico").sum()),
                                 pct((df["desenlace"]=="Crítico").sum(), total) + " del total", "neg"),
        ("Casos Leves",          fmt((df["desenlace"]=="Leve").sum()),
                                 pct((df["desenlace"]=="Leve").sum(), total) + " del total",   "pos"),
    ]
    for col, (label, value, delta, typ) in zip(cols2, kpis2):
        with col:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="label">{label}</div>
                    <div class="value">{value}</div>
                    <div class="delta delta-{typ}">{delta}</div>
                </div>""",
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────
# SECCIÓN 2 — ESTADÍSTICAS CUANTITATIVAS
# ─────────────────────────────────────────────
def render_quantitative(df: pd.DataFrame):
    st.markdown('<div class="section-header">📐 Estadísticas Cuantitativas</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Variable: Edad**")
        edad = df["edad"].dropna()
        kurt = stats.kurtosis(edad)
        skew = stats.skew(edad)
        rows = [
            ("N",            fmt(len(edad))),
            ("Media",        f"{edad.mean():.2f}"),
            ("Mediana",      f"{edad.median():.2f}"),
            ("Moda",         f"{edad.mode().iloc[0]}"),
            ("Desv. Est.",   f"{edad.std():.2f}"),
            ("Varianza",     f"{edad.var():.2f}"),
            ("Mín.",         f"{edad.min()}"),
            ("Máx.",         f"{edad.max()}"),
            ("Q1",           f"{edad.quantile(0.25):.2f}"),
            ("Q3",           f"{edad.quantile(0.75):.2f}"),
            ("IQR",          f"{edad.quantile(0.75)-edad.quantile(0.25):.2f}"),
            ("Asimetría",    f"{skew:.4f}"),
            ("Curtosis",     f"{kurt:.4f}"),
        ]
        html = '<table class="stat-table"><tr><th>Estadístico</th><th>Valor</th></tr>'
        for r in rows:
            html += f"<tr><td>{r[0]}</td><td>{r[1]}</td></tr>"
        html += "</table>"
        st.markdown(html, unsafe_allow_html=True)

    with col2:
        st.markdown("**Variable: Días Hospitalizado**")
        dias = df["dias_hospitalizado"].dropna()
        if len(dias) > 0:
            kurt2 = stats.kurtosis(dias)
            skew2 = stats.skew(dias)
            rows2 = [
                ("N (hospitalizados)", fmt(len(dias))),
                ("Media",        f"{dias.mean():.2f}"),
                ("Mediana",      f"{dias.median():.2f}"),
                ("Moda",         f"{dias.mode().iloc[0]}"),
                ("Desv. Est.",   f"{dias.std():.2f}"),
                ("Varianza",     f"{dias.var():.2f}"),
                ("Mín.",         f"{dias.min():.1f}"),
                ("Máx.",         f"{dias.max():.1f}"),
                ("Q1",           f"{dias.quantile(0.25):.2f}"),
                ("Q3",           f"{dias.quantile(0.75):.2f}"),
                ("IQR",          f"{dias.quantile(0.75)-dias.quantile(0.25):.2f}"),
                ("Asimetría",    f"{skew2:.4f}"),
                ("Curtosis",     f"{kurt2:.4f}"),
            ]
            html2 = '<table class="stat-table"><tr><th>Estadístico</th><th>Valor</th></tr>'
            for r in rows2:
                html2 += f"<tr><td>{r[0]}</td><td>{r[1]}</td></tr>"
            html2 += "</table>"
            st.markdown(html2, unsafe_allow_html=True)
        else:
            st.info("Sin datos de hospitalización en la selección.")


# ─────────────────────────────────────────────
# SECCIÓN 3 — ESTADÍSTICAS CUALITATIVAS
# ─────────────────────────────────────────────
def render_qualitative(df: pd.DataFrame):
    st.markdown('<div class="section-header">🏷️ Estadísticas Cualitativas</div>', unsafe_allow_html=True)

    cols = st.columns(4)
    cat_vars = [
        ("Desenlace",     "desenlace"),
        ("Resultado PCR", "resultado_pcr"),
        ("Género",        "genero"),
        ("Variante",      "variante"),
    ]
    pill_map = {
        "Leve": "green", "Grave": "orange", "Crítico": "orange",
        "Fallecido": "red", "Positivo": "blue", "Negativo": "red",
        "Masculino": "blue", "Femenino": "red", "Otro": "blue",
        "Original": "green", "Alpha": "blue", "Delta": "orange",
        "Ómicron": "orange", "Otra": "blue",
    }

    for col, (title, var) in zip(cols, cat_vars):
        with col:
            st.markdown(f"**{title}**")
            vc = df[var].value_counts()
            html = '<table class="stat-table"><tr><th>Categoría</th><th>N</th><th>%</th></tr>'
            for cat, cnt in vc.items():
                pc  = cnt / len(df) * 100
                cls = pill_map.get(cat, "blue")
                html += (
                    f'<tr><td><span class="pill pill-{cls}">{cat}</span></td>'
                    f"<td>{fmt(cnt)}</td><td>{pc:.1f}%</td></tr>"
                )
            html += "</table>"
            st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SECCIÓN 4 — ANÁLISIS GRÁFICO DINÁMICO
# ─────────────────────────────────────────────
def render_charts(df: pd.DataFrame):
    st.markdown('<div class="section-header">📈 Análisis Gráfico Dinámico</div>', unsafe_allow_html=True)

    # ── 4.1 Serie temporal de casos ────────────────────────────────────
    st.markdown("#### Evolución Temporal de Casos por Desenlace")
    df_ts = (
        df.set_index("fecha_diagnostico")
        .groupby([pd.Grouper(freq="ME"), "desenlace"])
        .size()
        .reset_index(name="casos")
    )
    df_ts.columns = ["fecha", "desenlace", "casos"]
    fig_ts = px.line(
        df_ts, x="fecha", y="casos", color="desenlace",
        color_discrete_map=PALETTE,
        line_shape="spline",
        markers=True,
        labels={"fecha": "Fecha", "casos": "Casos", "desenlace": "Desenlace"},
    )
    fig_ts.update_traces(line_width=2, marker_size=4)
    fig_ts.update_layout(**PLOT_LAYOUT, height=380,
                         xaxis=dict(gridcolor="#21262d"),
                         yaxis=dict(gridcolor="#21262d"))
    st.plotly_chart(fig_ts, use_container_width=True)

    # ── 4.2 Pirámide de edad por género ────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Pirámide de Edad por Género")
        bins = list(range(0, 101, 10))
        labels = [f"{b}-{b+9}" for b in bins[:-1]]
        df["grupo_edad"] = pd.cut(df["edad"], bins=bins, labels=labels, right=False)
        piramide = (
            df[df["genero"].isin(["Masculino", "Femenino"])]
            .groupby(["grupo_edad", "genero"], observed=True)
            .size()
            .reset_index(name="casos")
        )
        piramide.loc[piramide["genero"] == "Masculino", "casos"] *= -1
        fig_pir = px.bar(
            piramide, x="casos", y="grupo_edad", color="genero",
            orientation="h",
            color_discrete_map=PALETTE,
            labels={"casos": "Casos", "grupo_edad": "Grupo Edad"},
        )
        fig_pir.update_layout(
            **PLOT_LAYOUT, height=400,
            xaxis=dict(
                gridcolor="#21262d",
                tickvals=[-400, -200, 0, 200, 400],
                ticktext=["400", "200", "0", "200", "400"],
            ),
            yaxis=dict(gridcolor="#21262d"),
            bargap=0.1,
        )
        st.plotly_chart(fig_pir, use_container_width=True)

    with col2:
        st.markdown("#### Distribución de Desenlace por Variante")
        df_var = (
            df.groupby(["variante", "desenlace"])
            .size()
            .reset_index(name="casos")
        )
        fig_var = px.bar(
            df_var, x="variante", y="casos", color="desenlace",
            barmode="stack",
            color_discrete_map=PALETTE,
            labels={"variante": "Variante", "casos": "Casos", "desenlace": "Desenlace"},
        )
        fig_var.update_layout(
            **PLOT_LAYOUT, height=400,
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
        )
        st.plotly_chart(fig_var, use_container_width=True)

    # ── 4.3 Mapa de calor — Departamento × Variante ─────────────────────
    st.markdown("#### Mapa de Calor: Casos por Departamento y Variante")
    pivot = (
        df.groupby(["departamento", "variante"])
        .size()
        .unstack(fill_value=0)
    )
    fig_heat = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="Blues",
        text=pivot.values,
        texttemplate="%{text:,}",
        showscale=True,
        colorbar=dict(tickfont=dict(color="#c9d1d9")),
    ))
    fig_heat.update_layout(
        **PLOT_LAYOUT, height=400,
        xaxis=dict(title="Variante", gridcolor="#21262d"),
        yaxis=dict(title="Departamento", gridcolor="#21262d"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── 4.4 Box-plot edad por desenlace y Scatter edad vs días hosp ──────
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Box-Plot: Edad por Desenlace")
        fig_box = px.box(
            df, x="desenlace", y="edad", color="desenlace",
            color_discrete_map=PALETTE,
            labels={"desenlace": "Desenlace", "edad": "Edad"},
            notched=True,
        )
        fig_box.update_layout(
            **PLOT_LAYOUT, height=380, showlegend=False,
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with col4:
        st.markdown("#### Dispersión: Edad vs Días Hospitalizado")
        df_hosp = df[df["dias_hospitalizado"].notna()].sample(
            min(2000, df["dias_hospitalizado"].notna().sum()), random_state=1
        )
        fig_scat = px.scatter(
            df_hosp, x="edad", y="dias_hospitalizado", color="desenlace",
            color_discrete_map=PALETTE,
            opacity=0.55,
            trendline="lowess",
            labels={"edad": "Edad", "dias_hospitalizado": "Días Hospitalizado",
                    "desenlace": "Desenlace"},
        )
        fig_scat.update_layout(
            **PLOT_LAYOUT, height=380,
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
        )
        st.plotly_chart(fig_scat, use_container_width=True)

    # ── 4.5 Treemap departamento / desenlace ────────────────────────────
    st.markdown("#### Treemap: Casos por Departamento y Desenlace")
    df_tree = (
        df.groupby(["departamento", "desenlace"])
        .size()
        .reset_index(name="casos")
    )
    fig_tree = px.treemap(
        df_tree,
        path=["departamento", "desenlace"],
        values="casos",
        color="desenlace",
        color_discrete_map=PALETTE,
    )
    fig_tree.update_layout(**PLOT_LAYOUT, height=420)
    fig_tree.update_traces(textfont=dict(color="#e6edf3"))
    st.plotly_chart(fig_tree, use_container_width=True)

    # ── 4.6 Histograma + KDE edad ────────────────────────────────────────
    col5, col6 = st.columns(2)
    with col5:
        st.markdown("#### Histograma de Edad")
        fig_hist = px.histogram(
            df, x="edad", nbins=40, color_discrete_sequence=["#58a6ff"],
            marginal="box",
            labels={"edad": "Edad", "count": "Frecuencia"},
        )
        fig_hist.update_layout(
            **PLOT_LAYOUT, height=380,
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col6:
        st.markdown("#### Violín: Días Hospitalizado por Variante")
        df_viol = df[df["dias_hospitalizado"].notna()]
        fig_viol = px.violin(
            df_viol, x="variante", y="dias_hospitalizado", color="variante",
            box=True, points=False,
            color_discrete_map=PALETTE,
            labels={"variante": "Variante", "dias_hospitalizado": "Días Hospitalizado"},
        )
        fig_viol.update_layout(
            **PLOT_LAYOUT, height=380, showlegend=False,
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
        )
        st.plotly_chart(fig_viol, use_container_width=True)

    # ── 4.7 Sunburst resultado PCR → desenlace ──────────────────────────
    st.markdown("#### Sunburst: PCR → Desenlace")
    df_sun = (
        df.groupby(["resultado_pcr", "desenlace"])
        .size()
        .reset_index(name="casos")
    )
    fig_sun = px.sunburst(
        df_sun,
        path=["resultado_pcr", "desenlace"],
        values="casos",
        color="desenlace",
        color_discrete_map=PALETTE,
    )
    fig_sun.update_layout(**PLOT_LAYOUT, height=460)
    st.plotly_chart(fig_sun, use_container_width=True)


# ─────────────────────────────────────────────
# SECCIÓN 5 — DATOS CRUDOS
# ─────────────────────────────────────────────
def render_raw_data(df: pd.DataFrame):
    st.markdown('<div class="section-header">🗃️ Datos Crudos</div>', unsafe_allow_html=True)
    with st.expander("Ver tabla de datos (primeros 500 registros)"):
        display_cols = [
            "fecha_diagnostico", "edad", "genero", "departamento",
            "variante", "dias_hospitalizado", "resultado_pcr", "desenlace",
        ]
        st.dataframe(
            df[display_cols].head(500),
            use_container_width=True,
            height=400,
        )
    csv = df[[
        "fecha_diagnostico", "edad", "genero", "departamento",
        "variante", "dias_hospitalizado", "resultado_pcr", "desenlace",
    ]].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar CSV completo",
        data=csv,
        file_name="covid_datos_sinteticos.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # Hero banner
    st.markdown(
        """<div class="hero-banner">
            <p class="hero-title">🦠 COVID-19 Analytics Dashboard</p>
            <p class="hero-sub">10 000 registros sintéticos · 8 variables · Análisis exploratorio interactivo</p>
        </div>""",
        unsafe_allow_html=True,
    )

    # Generar datos
    df_full = generate_covid_data(10_000)

    # Filtros en sidebar → df filtrado
    df = build_sidebar(df_full)

    if df.empty:
        st.warning("⚠️ No hay datos con los filtros seleccionados. Ajusta la selección.")
        return

    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 KPIs",
        "📐 Estadísticas",
        "📈 Gráficas",
        "🗃️ Datos",
    ])

    with tab1:
        render_kpis(df, df_full)

    with tab2:
        render_quantitative(df)
        st.markdown("---")
        render_qualitative(df)

    with tab3:
        render_charts(df)

    with tab4:
        render_raw_data(df)


if __name__ == "__main__":
    main()
