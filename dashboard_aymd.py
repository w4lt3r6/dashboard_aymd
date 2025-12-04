
# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

# ---------------------------------------------------------------------
# Configuración visual
# ---------------------------------------------------------------------
st.set_page_config(page_title="Dashboard AYMD", layout="wide")
sns.set(style="whitegrid")

# ---------------------------------------------------------------------
# Sidebar: Tuning rápido
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Tuning rápido (XGBoost)")
    eta = st.slider("Learning rate (eta)", 0.01, 0.15, 0.04, 0.01)
    max_depth = st.slider("Profundidad máxima (max_depth)", 3, 9, 5, 1)
    n_rounds = st.slider("Num Boost Rounds", 400, 2000, 1200, 50)
    lambda_l2 = st.slider("Regularización L2 (lambda)", 0.0, 5.0, 2.0, 0.5)
    min_child_weight = st.slider("min_child_weight", 1.0, 10.0, 3.0, 0.5)
    early_stop_rounds = st.slider("Early stopping rounds", 30, 200, 70, 10)
    st.caption("Ajusta y observa cómo cambian las métricas. El modelo se re-entrena al mover los sliders.")

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

# Normalización de texto
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

# ---------------- Mes cíclico (captura continuidad dic-ene) --------------
df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12.0)
df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12.0)

# ---------------------------------------------------------------------
# Segmentación (Tab 1) - KMeans por ciudad y total
# ---------------------------------------------------------------------
client_sales = (
    df.groupby(['Nombre cliente', 'Ciudad'], as_index=False)['Total']
      .sum()
)
client_sales['Ciudad_code'] = client_sales['Ciudad'].astype('category').cat.codes

# KMeans (n_init explícito)
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

# One-hot encoding de categóricas (quitamos 'mes' one-hot: usamos mes_sin/mes_cos)
df_encoded = pd.get_dummies(
    df_model[['Nombre cliente', 'Ciudad', 'trimestre']],
    columns=['Nombre cliente', 'Ciudad', 'trimestre'],
    dtype=float
)

# Numéricas agregadas (RFM) + mes cíclico
numericas = df_model[['total_cliente', 'freq_compra', 'ticket_promedio',
                      'std_ticket', 'recencia_dias', 'mes_sin', 'mes_cos']].copy()

# Relleno de NaN en numéricas con la mediana
for c in numericas.columns:
    if numericas[c].isna().any():
        numericas[c] = numericas[c].fillna(numericas[c].median())

# X final y saneo
X = pd.concat([df_encoded.reset_index(drop=True),
               numericas.reset_index(drop=True)], axis=1)
X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

# y (objetivo): transformamos con log1p para estabilizar
y = df_model['Total'].astype(float)
y = y.replace([np.inf, -np.inf], np.nan).fillna(y.median())
y_log = np.log1p(y)  # log(1 + y) para entrenamiento

# ---------------------------------------------------------------------
# VALIDACIÓN TEMPORAL (train antes de 1/oct/2023; test desde 1/oct/2023)
# ---------------------------------------------------------------------
cutoff = pd.to_datetime("2023-10-01")
train_idx = df_model['Fecha'] < cutoff
test_idx  = df_model['Fecha'] >= cutoff

# Seguridad: si por algún motivo no hay test o train, usamos split aleatorio
use_random_split = (test_idx.sum() == 0 or train_idx.sum() == 0)
if use_random_split:
    st.warning("No hay suficientes datos para validación temporal; usando split aleatorio 80/20.")
    X_train_df, X_test_df, y_train_s, y_test_s = train_test_split(
        X, y_log, test_size=0.20, random_state=42
    )
    # Para panel de residuos: usaremos índices del DataFrame
    test_indices = X_test_df.index
else:
    X_train_df, X_test_df = X.loc[train_idx], X.loc[test_idx]
    y_train_s, y_test_s   = y_log.loc[train_idx], y_log.loc[test_idx]
    test_indices = X_test_df.index  # índice original de df_model

# Convertir a float32 contiguo
X_train = np.ascontiguousarray(X_train_df.values, dtype=np.float32)
X_test  = np.ascontiguousarray(X_test_df.values,  dtype=np.float32)
y_train = np.ascontiguousarray(y_train_s.values,  dtype=np.float32)
y_test_log = np.ascontiguousarray(y_test_s.values, dtype=np.float32)

# DMatrix con nombres de features (garantiza mapeo consistente)
feature_names = list(X.columns)
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dvalid = xgb.DMatrix(X_test,  label=y_test_log, feature_names=feature_names)

# ---------------------------------------------------------------------
# Booster (API nativa) con parámetros del sidebar y early stopping
# ---------------------------------------------------------------------
params = {
    "objective": "reg:squarederror",  # sobre y_log
    "eval_metric": "rmse",
    "eta": float(eta),                 # learning_rate
    "max_depth": int(max_depth),
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "lambda": float(lambda_l2),        # L2
    "min_child_weight": float(min_child_weight),
    "tree_method": "hist",
    "nthread": 1,
    "seed": 42
}

evals = [(dtrain, "train"), (dvalid, "valid")]

booster = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=int(n_rounds),
    evals=evals,
    early_stopping_rounds=int(early_stop_rounds),
    verbose_eval=False
)

# ---------------------------------------------------------------------
# UI - Pestañas
# ---------------------------------------------------------------------
tab1, tab2 = st.tabs(["Segmentación de Clientes", "Predicción + Análisis de Errores"])

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
    st.title("Predicción del Monto + Análisis de Errores")

    # ----------------- Métricas del modelo (en pesos) -----------------
    y_pred_log = booster.predict(dvalid)  # predicciones en escala log
    y_pred = np.expm1(y_pred_log)         # invertimos a pesos
    y_test = np.expm1(y_test_log)         # valores reales en pesos

    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    st.subheader("Evaluación del modelo")
    st.write(f"**MAE (en pesos):** ${mean_absolute_error(y_test, y_pred):,.2f}")
    st.write(f"**RMSE (en pesos):** ${rmse:,.2f}")
    st.write(f"**R² (en pesos):** {r2_score(y_test, y_pred):.2f}")

    if use_random_split:
        st.caption("Split aleatorio 80/20 (no hubo suficientes filas para validación temporal).")
    else:
        st.caption("Validación temporal (train < 2023-10-01; test ≥ 2023-10-01). "
                   "Objetivo entrenado en log(1 + Total); métricas reportadas en pesos.")

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

    # Mes cíclico para el cliente
    mes_sin_pred = float(np.sin(2 * np.pi * mes_pred / 12.0))
    mes_cos_pred = float(np.cos(2 * np.pi * mes_pred / 12.0))

    # Construir vector de entrada (one-hot + RFM + mes cíclico) consistente con feature_names
    input_dict = {
        f"Nombre cliente_{cliente_input}": 1.0,
        f"Ciudad_{ciudad_pred}": 1.0,
        f"trimestre_{tri_pred}": 1.0,
        "total_cliente": float(agg_row['total_cliente']),
        "freq_compra": float(agg_row['freq_compra']),
        "ticket_promedio": float(agg_row['ticket_promedio']),
        "std_ticket": float(agg_row['std_ticket']),
        "recencia_dias": float(agg_row['recencia_dias']),
        "mes_sin": mes_sin_pred,
        "mes_cos": mes_cos_pred,
    }

    input_vector = pd.DataFrame([input_dict]).reindex(columns=feature_names, fill_value=0.0)
    input_np = np.ascontiguousarray(input_vector.values, dtype=np.float32)
    dinput = xgb.DMatrix(input_np, feature_names=feature_names)

    # Predicción en escala log y conversión a pesos
    predicted_total_log = float(booster.predict(dinput)[0])
    predicted_total = float(np.expm1(predicted_total_log))
    st.write(f"### Monto estimado: ${predicted_total:,.2f}")

    # ----------------- Análisis de residuos -----------------
    st.subheader("Análisis de errores (residuos)")
    # Residuos y DataFrame de evaluación (en pesos)
    residuos = y_test - y_pred
    df_eval = pd.DataFrame({
        "y_real": y_test,
        "y_pred": y_pred,
        "residuo": residuos
    }, index=test_indices)

    # Añadimos contexto (cliente, ciudad, fecha, total real del df_model)
    cols_ctx = ['Nombre cliente', 'Ciudad', 'Fecha', 'Total']
    ctx = df_model.loc[test_indices, cols_ctx].copy()
    df_eval_ctx = ctx.join(df_eval)

    st.markdown("**Top 15 mayores errores absolutos (en pesos)**")
    top_err = df_eval_ctx.assign(error_abs=lambda d: d["residuo"].abs()) \
                         .sort_values("error_abs", ascending=False) \
                         .head(15)
    st.dataframe(top_err)

    st.markdown("**Histograma de residuos (y_real - y_pred)**")
    fig_res, ax_res = plt.subplots(figsize=(8, 4))
    sns.histplot(df_eval_ctx["residuo"], bins=40, kde=True, ax=ax_res)
    ax_res.set_title("Distribución de residuos")
    ax_res.set_xlabel("Residuo (pesos)")
    st.pyplot(fig_res)

    # ----------------- Importancia de variables -----------------
    st.subheader("Importancia de variables (gain)")
    score = booster.get_score(importance_type='gain')  # dict: {feature_name: score}
    if len(score) == 0:
        st.info("La importancia de variables no está disponible en este booster.")
    else:
        importance_df = pd.DataFrame({
            "Feature": list(score.keys()),
            "Importance": list(score.values())
        }).sort_values(by="Importance", ascending=False)

        fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
        sns.barplot(data=importance_df.head(20), x="Importance", y="Feature", ax=ax_imp)
        ax_imp.set_title("Top 20 variables más importantes (gain)")
        st.pyplot(fig_imp)

    # ----------------- Guardar / Descargar modelo -----------------
    st.subheader("Guardar modelo entrenado")
    raw = booster.save_raw(raw_format=True)
    st.download_button(
        label="⬇️ Descargar booster (XGBoost) .bin",
        data=raw,
        file_name="modelo_xgb_aymd.bin",
        mime="application/octet-stream"
    )

    # ----------------- Diagnóstico rápido -----------------
    with st.expander("Diagnóstico (si algo falla)"):
        st.write("Filas train:", int(train_idx.sum()) if 'train_idx' in locals() and not use_random_split else "Split aleatorio")
        st.write("Filas test:", int(test_idx.sum()) if 'test_idx' in locals() and not use_random_split else "Split aleatorio")
        st.write("X shape:", X.shape)
        st.write("y (Total) shape:", y.shape)
        st.write("¿Hay NaN en X?", np.isnan(X.values).any())
        st.write("¿Hay NaN en y?", np.isnan(y.values).any())
        st.write("¿Hay Inf en X?", np.isinf(X.values).any())
        st.write("¿Hay Inf en y?", np.isinf(y.values).any())
        st.write("Primeras 10 columnas de X:", list(X.columns[:10]))
        st.write("Mejor iteración (early stopping):", booster.best_iteration)
