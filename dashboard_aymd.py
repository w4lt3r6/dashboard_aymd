
# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# ---------------------------------------------------------------------
# Configuración visual
# ---------------------------------------------------------------------
st.set_page_config(page_title="Dashboard AYMD", layout="wide")
sns.set(style="whitegrid")

# ---------------------------------------------------------------------
# Helper para fechas (maneja strings y seriales de Excel)
# ---------------------------------------------------------------------
def parse_fecha(series: pd.Series) -> pd.Series:
    s = series.copy()
    # 1) Parseo estándar con día primero (formato latino)
    s_dt = pd.to_datetime(s, errors='coerce', dayfirst=True)

    # 2) Donde quedó NaT y el original es numérico, asumir serial Excel (origen 1899-12-30)
    mask_num = s_dt.isna() & s.apply(lambda x: isinstance(x, (int, float)))
    if mask_num.any():
        s_dt.loc[mask_num] = pd.to_datetime(s[mask_num], unit='D', origin='1899-12-30', errors='coerce')
    return s_dt

# ---------------------------------------------------------------------
# Cargar datos
# ---------------------------------------------------------------------
df = pd.read_excel("Base-2023-AYMD-OpenRefine (1) 2.xlsx", engine="openpyxl")

# Normalización básica de texto (evita problemas de espacios y mayúsculas)
for col in ['Ciudad', 'Vendedor', 'Nombre cliente']:
    df[col] = df[col].astype(str).str.strip().str.lower()

# Fechas robustas
df['Fecha'] = parse_fecha(df['Fecha'])

# Derivados de tiempo por fila
mes_series = df['Fecha'].dt.month
tri_series = ((mes_series - 1) // 3 + 1)

# Rellenos con moda si faltan
if mes_series.isna().any():
    moda_mes = mes_series.dropna().mode()
    moda_mes = int(moda_mes.iloc[0]) if len(moda_mes) else 1
    mes_series = mes_series.fillna(moda_mes)

if tri_series.isna().any():
    moda_tri = tri_series.dropna().mode()
    moda_tri = int(moda_tri.iloc[0]) if len(moda_tri) else 1
    tri_series = tri_series.fillna(moda_tri)

df['mes'] = mes_series.astype(int)
df['trimestre'] = tri_series.astype(int)

# ---------------------------------------------------------------------
# Segmentación (Tab 1) - KMeans por ciudad y total
# ---------------------------------------------------------------------
client_sales = (
    df.groupby(['Nombre cliente', 'Ciudad'], as_index=False)['Total']
      .sum()
)
client_sales['Ciudad_code'] = client_sales['Ciudad'].astype('category').cat.codes

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
client_sales['Cluster'] = kmeans.fit_predict(client_sales[['Ciudad_code', 'Total']])

# ---------------------------------------------------------------------
# Agregados RFM a nivel cliente
# ---------------------------------------------------------------------
agg = df.groupby('Nombre cliente').agg(
    total_cliente=('Total', 'sum'),
    freq_compra=('Total', 'count'),
    ticket_promedio=('Total', 'mean'),
    std_ticket=('Total', 'std'),
    ultima_fecha=('Fecha', 'max')
).reset_index()

# Ciudad principal del cliente (modo del histórico)
ciudad_principal = (
    df.groupby('Nombre cliente')['Ciudad']
      .agg(lambda s: s.value_counts().index[0] if len(s) else np.nan)
      .reset_index()
      .rename(columns={'Ciudad': 'ciudad_principal'})
)
agg = agg.merge(ciudad_principal, on='Nombre cliente', how='left')

# Recencia en días (hasta la última fecha de la base)
fecha_ref = df['Fecha'].max()
agg['recencia_dias'] = (fecha_ref - agg['ultima_fecha']).dt.days

# Rellenos de NaN en agregados
agg['std_ticket'] = agg['std_ticket'].fillna(0)
agg['recencia_dias'] = agg['recencia_dias'].fillna(agg['recencia_dias'].median())

# Mes y trimestre del último registro del cliente (para predicción por cliente)
agg['mes_ultimo'] = agg['ultima_fecha'].dt.month
# Si faltó, usa la moda global
if agg['mes_ultimo'].isna().any():
    moda_mes_global = df['mes'].mode()
    moda_mes_global = int(moda_mes_global.iloc[0]) if len(moda_mes_global) else 1
    agg['mes_ultimo'] = agg['mes_ultimo'].fillna(moda_mes_global)
agg['tri_ultimo'] = ((agg['mes_ultimo'] - 1) // 3 + 1).astype(int)

# ---------------------------------------------------------------------
# Construcción del dataset de modelado (robusto)
# ---------------------------------------------------------------------
df_model = df.merge(
    agg[['Nombre cliente', 'ciudad_principal', 'total_cliente',
         'freq_compra', 'ticket_promedio', 'std_ticket',
         'recencia_dias', 'mes_ultimo', 'tri_ultimo']],
    on='Nombre cliente', how='left'
)

# One-hot encoding de categóricas (fila y tiempo)
df_encoded = pd.get_dummies(
    df_model[['Nombre cliente', 'Ciudad', 'mes', 'trimestre']],
    columns=['Nombre cliente', 'Ciudad', 'mes', 'trimestre'],
    dtype=float  # fuerza columnas numéricas
)

# Numéricas agregadas (RFM)
numericas = df_model[['total_cliente', 'freq_compra', 'ticket_promedio',
                      'std_ticket', 'recencia_dias']].copy()

# Relleno de NaN en numéricas con la mediana
for c in numericas.columns:
    if numericas[c].isna().any():
        numericas[c] = numericas[c].fillna(numericas[c].median())

# X final y saneo
X = pd.concat([df_encoded.reset_index(drop=True),
               numericas.reset_index(drop=True)], axis=1)

# Sustituye Inf/-Inf y NaN, fuerza float
X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

# y (objetivo)
y = df_model['Total'].astype(float)
y = y.replace([np.inf, -np.inf], np.nan).fillna(y.median())

# Split con numpy arrays (evita problemas de índices en xgboost)
X_train, X_test, y_train, y_test = train_test_split(
    X.values,
    y.values,
    test_size=0.20,
    random_state=42
)

# ---------------------------------------------------------------------
# Modelo XGBoost con early stopping y métrica explícita
# ---------------------------------------------------------------------
model = xgb.XGBRegressor(
    random_state=42,
    verbosity=0,
    n_estimators=2000,        # más árboles pero frenados por early stopping
    max_depth=6,
    learning_rate=0.03,       # tasa baja para generalizar
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.0,
    reg_lambda=1.0,
    tree_method='hist'
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='rmse',
    verbose=False,
    early_stopping_rounds=100
)

# ---------------------------------------------------------------------
# UI - Pestañas
# ---------------------------------------------------------------------
tab1, tab2 = st.tabs(["Segmentación de Clientes", "Predicción con XGBoost (Mejorado)"])

# ============================== TAB 1 ==============================
with tab1:
    st.title("Segmentación de Clientes AYMD")
    st.write("Segmentación por ciudad y monto total de compras usando K-Means.")

    # Selector de cluster
    selected_cluster = st.selectbox(
        "Selecciona un cluster:", sorted(client_sales['Cluster'].unique())
    )
    filtered_data = client_sales[client_sales['Cluster'] == selected_cluster]

    st.subheader("Clientes en el cluster seleccionado")
    st.dataframe(filtered_data[['Nombre cliente', 'Ciudad', 'Total']])

    st.subheader("Distribución de clientes por ciudad y monto total")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=client_sales, x='Ciudad_code', y='Total',
        hue='Cluster', palette='Set2', ax=ax
    )
    ax.set_title("Segmentación de clientes")
    ax.set_xlabel("Código de ciudad")
    ax.set_ylabel("Total de compras")
    st.pyplot(fig)

    st.subheader("Matriz de correlación")
    corr_matrix = client_sales[['Ciudad_code', 'Total', 'Cluster']].corr()
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
    st.pyplot(fig_corr)

    st.subheader("Distribución de variables numéricas")
    for col in ['Ciudad_code', 'Total', 'Cluster']:
        fig_dist, ax_dist = plt.subplots()
        sns.histplot(client_sales[col], kde=True, ax=ax_dist)
        ax_dist.set_title(f"Distribución de {col}")
        st.pyplot(fig_dist)

    st.subheader("Relaciones entre variables numéricas")
    pairplot_fig = sns.pairplot(
        client_sales[['Ciudad_code', 'Total', 'Cluster']],
        hue='Cluster', palette='Set2'
    )
    st.pyplot(pairplot_fig.fig)

    st.subheader("Diagramas de bigotes y violín")
    for col in ['Ciudad_code', 'Total']:
        fig_box, ax_box = plt.subplots()
        sns.boxplot(data=client_sales, x='Cluster', y=col, ax=ax_box)
        ax_box.set_title(f"Boxplot de {col}")
        st.pyplot(fig_box)

        fig_violin, ax_violin = plt.subplots()
        sns.violinplot(data=client_sales, x='Cluster', y=col, palette='Set2', ax=ax_violin)
        ax_violin.set_title(f"Violinplot de {col}")
        st.pyplot(fig_violin)

# ============================== TAB 2 ==============================
with tab2:
    st.title("Predicción del Monto de Compra con XGBoost (Mejorado)")

    # ----------------- Métricas del modelo -----------------
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.subheader("Evaluación del modelo")
    st.write(f"**MAE:** ${mean_absolute_error(y_test, y_pred):,.2f}")
    st.write(f"**RMSE:** ${rmse:,.2f}")
    st.write(f"**R²:** {r2_score(y_test, y_pred):.2f}")
    st.caption(
        "El modelo usa ciudad, mes, trimestre y agregados RFM por cliente; entrenado con early stopping."
    )

    # ----------------- Predicción personalizada (solo Cliente) -----------------
    st.subheader("Predicción personalizada")

    clientes = sorted(df['Nombre cliente'].dropna().unique())
    if len(clientes) == 0:
        st.warning("No hay clientes en la base para realizar predicciones.")
        st.stop()

    cliente_input = st.selectbox("Cliente:", clientes, key="sel_cliente")

    # Buscar agregados del cliente seleccionado (con fallbacks)
    if cliente_input not in agg['Nombre cliente'].values:
        st.warning("No se encontraron agregados para el cliente seleccionado.")
        st.stop()

    agg_row = agg[agg['Nombre cliente'] == cliente_input].iloc[0]

    # Inferir ciudad principal con fallback a la moda global si es NaN
    ciudad_pred = str(agg_row['ciudad_principal']) if pd.notna(agg_row['ciudad_principal']) else None
    if not ciudad_pred or ciudad_pred == 'nan':
        moda_ciudad_global = df['Ciudad'].mode()
        ciudad_pred = str(moda_ciudad_global.iloc[0]) if len(moda_ciudad_global) else 'mosquera'

    # Mes y trimestre del último registro del cliente (ya calculados; con fallback)
    mes_pred = int(agg_row['mes_ultimo']) if pd.notna(agg_row['mes_ultimo']) else int(df['mes'].mode().iloc[0])
    tri_pred = int(agg_row['tri_ultimo']) if pd.notna(agg_row['tri_ultimo']) else int(df['trimestre'].mode().iloc[0])

    # Construir vector de entrada (one-hot + RFM) consistente con X.columns
    input_dict = {
        f"Nombre cliente_{cliente_input}": 1.0,
        f"Ciudad_{ciudad_pred}": 1.0,
        f"mes_{mes_pred}": 1.0,
        f"trimestre_{tri_pred}": 1.0,
        "total_cliente": float(agg_row['total_cliente']),
        "freq_compra": float(agg_row['freq_compra']),
        "ticket_promedio": float(agg_row['ticket_promedio']),
        "std_ticket": float(agg_row['std_ticket']),
        "recencia_dias": float(agg_row['recencia_dias']),
    }

    input_vector = pd.DataFrame([input_dict]).reindex(columns=X.columns, fill_value=0.0)

    # Predicción (usar .values para mayor compatibilidad)
    predicted_total = float(model.predict(input_vector.values)[0])
    st.write(f"### Monto estimado: ${predicted_total:,.2f}")

    # ----------------- Importancia de variables -----------------
    st.subheader("Importancia de variables")
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
    sns.barplot(data=importance_df.head(20), x="Importance", y="Feature", ax=ax_imp)
    ax_imp.set_title("Top 20 variables más importantes")
    st.pyplot(fig_imp)

    # ----------------- Diagnóstico rápido -----------------
    with st.expander("Diagnóstico (si algo falla)"):
        st.write("X shape:", X.shape)
        st.write("y shape:", y.shape)
        st.write("¿Hay NaN en X?", np.isnan(X.values).any())
        st.write("¿Hay NaN en y?", np.isnan(y.values).any())
        st.write("¿Hay Inf en X?", np.isinf(X.values).any())
        st.write("¿Hay Inf en y?", np.isinf(y.values).any())
        st.write("Primeras 10 columnas de X:", list(X.columns[:10]))
        st.write("Mejor iteración del modelo:", getattr(model, "best_iteration", None))
