
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
# Cargar datos
# ---------------------------------------------------------------------
df = pd.read_excel("Base-2023-AYMD-OpenRefine (1) 2.xlsx", engine="openpyxl")

# Normalización básica de texto (evita problemas de espacios y mayúsculas)
for col in ['Ciudad', 'Vendedor', 'Nombre cliente']:
    df[col] = df[col].astype(str).str.strip().str.lower()

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
# Modelo XGBoost (tab 2) - usando SOLO "Nombre cliente"
# ---------------------------------------------------------------------
df_model = df.copy()

# One-hot encoding SOLO para Nombre cliente
df_encoded = pd.get_dummies(df_model[['Nombre cliente']])

X = df_encoded
y = df_model['Total'].astype(float)

# Train/Test split y entrenamiento
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = xgb.XGBRegressor(
    random_state=42,
    verbosity=0,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9
)
model.fit(X_train, y_train)

# ---------------------------------------------------------------------
# UI - Pestañas
# ---------------------------------------------------------------------
tab1, tab2 = st.tabs(["Segmentación de Clientes", "Predicción con XGBoost"])

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
    st.title("Predicción del Monto de Compra con XGBoost")

    # ----------------- Métricas del modelo -----------------
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.subheader("Evaluación del modelo")
    st.write(f"**MAE:** ${mean_absolute_error(y_test, y_pred):,.2f}")
    st.write(f"**RMSE:** ${rmse:,.2f}")
    st.write(f"**R²:** {r2_score(y_test, y_pred):.2f}")

    # ----------------- Predicción personalizada -----------------
    st.subheader("Predicción personalizada")

    # Fuente para la UI: df ya está normalizado
    clientes = sorted(df['Nombre cliente'].dropna().unique())
    if len(clientes) == 0:
        st.warning("No hay clientes en la base para realizar predicciones.")
        st.stop()

    cliente_input = st.selectbox("Cliente:", clientes, key="sel_cliente")

    # Construir vector de entrada (one-hot) consistente con X.columns
    input_dict = {f"Nombre cliente_{cliente_input}": 1}
    input_vector = pd.DataFrame([input_dict]).reindex(columns=X.columns, fill_value=0)

    # Predicción
    predicted_total = float(model.predict(input_vector)[0])
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
