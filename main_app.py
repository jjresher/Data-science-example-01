import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from faker import Faker

# ------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------
st.set_page_config(
    page_title="Dashboard COVID",
    page_icon="🦠",
    layout="wide"
)

fake = Faker()
np.random.seed(42)

# ------------------------------------------------
# GENERAR DATOS SINTÉTICOS
# ------------------------------------------------
@st.cache_data
def generar_datos():

    n = 10000

    ciudades = [
        "Bogotá", "Medellín", "Cali",
        "Barranquilla", "Cartagena"
    ]

    estados = [
        "Recuperado",
        "Hospitalizado",
        "UCI",
        "Fallecido"
    ]

    vacunas = [
        "Pfizer",
        "Moderna",
        "Sinovac",
        "AstraZeneca",
        "Ninguna"
    ]

    df = pd.DataFrame({
        "ID_Paciente": np.arange(1, n + 1),

        "Edad": np.random.randint(1, 95, n),

        "Genero": np.random.choice(
            ["Masculino", "Femenino"],
            n
        ),

        "Ciudad": np.random.choice(
            ciudades,
            n
        ),

        "Fecha_Diagnostico": pd.to_datetime(
            np.random.choice(
                pd.date_range(
                    "2020-01-01",
                    "2023-12-31"
                ),
                n
            )
        ),

        "Estado": np.random.choice(
            estados,
            n,
            p=[0.75, 0.15, 0.07, 0.03]
        ),

        "Tipo_Vacuna": np.random.choice(
            vacunas,
            n,
            p=[0.35, 0.25, 0.15, 0.15, 0.10]
        ),

        "Dias_Hospitalizacion": np.random.poisson(
            6,
            n
        )
    })

    return df


df = generar_datos()

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------
st.sidebar.title("Filtros")

ciudad = st.sidebar.multiselect(
    "Ciudad",
    df["Ciudad"].unique(),
    default=df["Ciudad"].unique()
)

estado = st.sidebar.multiselect(
    "Estado",
    df["Estado"].unique(),
    default=df["Estado"].unique()
)

df_filtrado = df[
    (df["Ciudad"].isin(ciudad)) &
    (df["Estado"].isin(estado))
]

# ------------------------------------------------
# TÍTULO
# ------------------------------------------------
st.title("🦠 Dashboard Analítico COVID")

st.markdown(
    "Datos sintéticos generados automáticamente."
)

# ------------------------------------------------
# MÉTRICAS CUANTITATIVAS
# ------------------------------------------------
st.header("📊 Métricas Cuantitativas")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Pacientes",
    f"{len(df_filtrado):,}"
)

col2.metric(
    "Edad Promedio",
    round(df_filtrado["Edad"].mean(), 1)
)

col3.metric(
    "Edad Máxima",
    df_filtrado["Edad"].max()
)

col4.metric(
    "Promedio Hospitalización",
    round(
        df_filtrado[
            "Dias_Hospitalizacion"
        ].mean(),
        1
    )
)

# ------------------------------------------------
# MÉTRICAS CUALITATIVAS
# ------------------------------------------------
st.header("📋 Métricas Cualitativas")

c1, c2 = st.columns(2)

with c1:
    st.write(
        "Estado más frecuente:"
    )
    st.success(
        df_filtrado["Estado"]
        .mode()[0]
    )

with c2:
    st.write(
        "Vacuna más frecuente:"
    )
    st.success(
        df_filtrado[
            "Tipo_Vacuna"
        ].mode()[0]
    )

# ------------------------------------------------
# GRÁFICAS DINÁMICAS
# ------------------------------------------------
st.header("📈 Análisis Gráfico")

col1, col2 = st.columns(2)

# 1. Casos por ciudad
fig1 = px.bar(
    df_filtrado
    .groupby("Ciudad")
    .size()
    .reset_index(name="Casos"),
    x="Ciudad",
    y="Casos",
    title="Casos por Ciudad"
)

col1.plotly_chart(
    fig1,
    use_container_width=True
)

# 2. Distribución estados
fig2 = px.pie(
    df_filtrado,
    names="Estado",
    title="Distribución Estado Paciente"
)

col2.plotly_chart(
    fig2,
    use_container_width=True
)

# ------------------------------------------------
# TENDENCIA TEMPORAL
# ------------------------------------------------
casos_fecha = (
    df_filtrado
    .groupby(
        pd.Grouper(
            key="Fecha_Diagnostico",
            freq="M"
        )
    )
    .size()
    .reset_index(name="Casos")
)

fig3 = px.line(
    casos_fecha,
    x="Fecha_Diagnostico",
    y="Casos",
    markers=True,
    title="Evolución Temporal de Casos"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# ------------------------------------------------
# HISTOGRAMA EDADES
# ------------------------------------------------
fig4 = px.histogram(
    df_filtrado,
    x="Edad",
    nbins=20,
    title="Distribución de Edades"
)

st.plotly_chart(
    fig4,
    use_container_width=True
)

# ------------------------------------------------
# DISPERSIÓN
# ------------------------------------------------
fig5 = px.scatter(
    df_filtrado,
    x="Edad",
    y="Dias_Hospitalizacion",
    color="Estado",
    title="Edad vs Hospitalización"
)

st.plotly_chart(
    fig5,
    use_container_width=True
)

# ------------------------------------------------
# TABLA
# ------------------------------------------------
st.header("📄 Datos")

st.dataframe(
    df_filtrado,
    use_container_width=True
)
