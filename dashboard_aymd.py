
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
# Helpers para manejar fechas con strings y seriales de Excel
# ---------------------------------------------------------------------
def parse_fecha(series: pd.Series) -> pd.Series:
    s = series.copy()
    # 1) Intento de parseo estándar con día primero (formato latino)
    s_dt = pd.to_datetime(s, errors='coerce', dayfirst=True)
    # 2) Si quedaron NaT y el valor original es numérico, asumir serial Excel
    #    Excel usa origen '1899-12-30' para días.
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
# Si alguna fecha es NaT, rellenamos mes/trimestre con la moda del conjunto
mes_series = df['Fecha'].dt.month
tri_series = ((mes_series - 1) // 3 + 1)
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
# Segmentación (tab 1) - KMeans por ciudad y total
# ---------------------------------------------------------------------
client_sales = (
    df.groupby(['Nombre cliente', 'Ciudad'], as_index=False)['Total']
      .sum()
)

# Codificar Ciudad para el scatter (solo para visualización/cluster)
client_sales['Ciudad_code'] = client_sales['Ciudad'].astype('category').cat.codes

# Clustering
X_cluster = client_sales[['Ciudad_code', 'Total']]
kmeans = KMeans(n_clusters=4, random_state=42)
client_sales['Cluster'] = kmeans.fit_predict(X_cluster)

# ---------------------------------------------------------------------
# Features agregadas a nivel cliente (RFM-like)
# ---------------------------------------------------------------------
# Frecuencia, total, ticket promedio, desviación del ticket y última fecha por cliente
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

# Recencia: días desde la última compra del cliente hasta la última fecha en la base
fecha_ref = df['Fecha'].max()
agg['recencia_dias'] = (fecha_ref - agg['ultima_fecha']).dt.days
# Rellenos por si hay NaN
agg['std_ticket'] = agg['std_ticket'].fillna(0)
agg['recencia_dias'] = agg['recencia_dias'].fillna(agg['recencia_dias'].median())

# Mes y trimestre del último registro del cliente (para predicción por cliente)
agg['mes_ultimo'] = agg['ultima_fecha'].dt.month.fillna(df['mes'].mode().iloc[0])
agg['tri_ultimo'] = ((agg['mes_ultimo'] - 1) // 3 + 1).astype(int)

# ---------------------------------------------------------------------
# Construcción del dataset de modelado (unimos agregados por cliente)
# ---------------------------------------------------------------------
df_model = df.merge(agg[['Nombre cliente', 'ciudad_principal', 'total_cliente',
                         'freq_compra', 'ticket_promedio', 'std_ticket',
                         'recencia_dias']],
                    on='Nombre cliente', how='left')

# One-hot encoding de categóricas (usamos variables de fila y agregadas)
# - Nombre cliente (clave principal de la UI)
# - Ciudad de la fila (mayor granularidad)
# - Mes y trimestre de la fila (estacionalidad)
df_encoded = pd.get_dummies(
    df_model[['Nombre cliente', 'Ciudad', 'mes', 'trimestre']],
    columns=['Nombre cliente', 'Ciudad', 'mes', 'trimestre']
)

# Variables numéricas agregadas (RFM)
numericas = df_model[['total_cliente', 'freq_compra', 'ticket_promedio',
                      'std_ticket', 'recencia_dias']].reset_index(drop=True)

# X final
X = pd.concat([df_encoded.reset_index(drop=True), numericas], axis=1)
y = df_model['Total'].astype(float)

# Train/Test split y entrenamiento con early stopping
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = xgb.XGBRegressor(
    random_state=42,
    verbosity=0,
    n_estimators=2000,        # más árboles pero frenados por early stopping
    max_depth=6,
    learning_rate=0.03,       # tasa más baja para generalizar mejor
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.0,
    reg_lambda=1.0,
    tree_method='hist'        # rápido y estable
)

# Early stopping: se detiene si no mejora 100 iteraciones
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
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
        "El modelo ahora usa ciudad, mes, trimestre y agregados RFM por cliente. "
        "Entrenado con early stopping para evitar sobreajuste."
    )

    # ----------------- Predicción personalizada (solo Cliente) -----------------
    st.subheader("Predicción personalizada")
    clientes = sorted(df['Nombre cliente'].dropna().unique())
    if len(clientes) == 0:
        st.warning("No hay clientes en la base para realizar predicciones.")
        st.stop()

    cliente_input = st.selectbox("Cliente:", clientes, key="sel_cliente")

    # Tomamos agregados del cliente seleccionado
    agg_row = agg[agg['Nombre cliente'] == cliente_input].iloc[0]

    # Inferimos ciudad principal y mes/trimestre del último registro del cliente
    ciudad_pred = str(agg_row['ciudad_principal'])
    mes_pred = int(agg_row['mes_ultimo'])
    tri_pred = int(agg_row['tri_ultimo'])

    # Construir vector de entrada (one-hot) consistente con X.columns
    input_dict = {
        # One-hot de cliente, ciudad, mes, trimestre
        f"Nombre cliente_{cliente_input}": 1,
        f"Ciudad_{ciudad_pred}": 1,
        f"mes_{mes_pred}": 1,
        f"trimestre_{tri_pred}": 1,
        # Numéricas agregadas
        "total_cliente": float(agg_row['total_cliente']),
        "freq_compra": float(agg_row['freq_compra']),
        "ticket_promedio": float(agg_row['ticket_promedio']),
        "std_ticket": float(agg_row['std_ticket']),
        "recencia_dias": float(agg_row['recencia_dias']),
    }

    # Vector con todas las columnas del entrenamiento
    input_vector = pd.DataFrame([input_dict]).reindex(columns=X.columns, fill_value=0)

    # Predicción
    predicted_total = float(model.predict(input_vector)[0])

    st.write(f"### Monto estimado: ${predicted_total:,.2f}")
    st.caption(
        "Para la predicción, se usó la ciudad principal y el mes/trimestre del último registro del cliente, "
        "además de sus agregados históricos (frecuencia, ticket promedio, total, desviación, recencia)."
    )

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
