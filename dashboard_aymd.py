import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Cargar datos
df = pd.read_excel("Base-2023-AYMD-OpenRefine (1) 2.xlsx", engine="openpyxl")
df['Ciudad'] = df['Ciudad'].str.strip().str.lower()

# Agrupar por cliente y ciudad
client_sales = df.groupby(['Nombre cliente', 'Ciudad'])['Total'].sum().reset_index()
client_sales['Ciudad_code'] = client_sales['Ciudad'].astype('category').cat.codes

# Clustering
X_cluster = client_sales[['Ciudad_code', 'Total']]
kmeans = KMeans(n_clusters=4, random_state=42)
client_sales['Cluster'] = kmeans.fit_predict(X_cluster)

# Tabs
tab1, tab2 = st.tabs(["Segmentación de Clientes", "Predicción con XGBoost"])

with tab1:
    st.title("Segmentación de Clientes AYMD")
    st.write("Segmentación por ciudad y monto total de compras usando K-Means.")

    selected_cluster = st.selectbox("Selecciona un cluster:", sorted(client_sales['Cluster'].unique()))
    filtered_data = client_sales[client_sales['Cluster'] == selected_cluster]

    st.subheader("Clientes en el cluster seleccionado")
    st.dataframe(filtered_data[['Nombre cliente', 'Ciudad', 'Total']])

    st.subheader("Distribución de clientes por ciudad y monto total")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.scatterplot(data=client_sales, x='Ciudad_code', y='Total', hue='Cluster', palette='Set2', ax=ax)
    ax.set_title("Segmentación de clientes")
    st.pyplot(fig)

    st.subheader("Matriz de correlación")
    corr_matrix = client_sales[['Ciudad_code', 'Total', 'Cluster']].corr()
    fig_corr, ax_corr = plt.subplots(figsize=(8,6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
    st.pyplot(fig_corr)

    st.subheader("Distribución de variables numéricas")
    for col in ['Ciudad_code', 'Total', 'Cluster']:
        fig_dist, ax_dist = plt.subplots()
        sns.histplot(client_sales[col], kde=True, ax=ax_dist)
        ax_dist.set_title(f"Distribución de {col}")
        st.pyplot(fig_dist)

    st.subheader("Relaciones entre variables numéricas")
    pairplot_fig = sns.pairplot(client_sales[['Ciudad_code', 'Total', 'Cluster']], hue='Cluster', palette='Set2')
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

with tab2:
    st.title("Predicción del Monto de Compra con XGBoost")

    # Codificación
    df_model = df.copy()
    for col in ['Ciudad', 'Vendedor', 'Nombre cliente']:
        df_model[col] = df_model[col].str.strip().str.lower()
    df_encoded = pd.get_dummies(df_model[['Ciudad', 'Vendedor', 'Nombre cliente']])
    X = df_encoded
    y = df_model['Total']

    # División y entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(random_state=42, verbosity=0)
    model.fit(X_train, y_train)

    # Métricas
    y_pred = model.predict(X_test)
    st.subheader("Evaluación del modelo")
    st.write(f"**MAE:** ${mean_absolute_error(y_test, y_pred):,.2f}")
    st.write(f"**RMSE:** ${mean_squared_error(y_test, y_pred, squared=False):,.2f}")
    st.write(f"**R²:** {r2_score(y_test, y_pred):.2f}")

    # Predicción personalizada
    st.subheader("Predicción personalizada")
    ciudad_input = st.selectbox("Ciudad:", sorted(df['Ciudad'].unique()))
    vendedor_input = st.selectbox("Vendedor:", sorted(df['Vendedor'].unique()))
    cliente_input = st.selectbox("Cliente:", sorted(df['Nombre cliente'].unique()))

    input_dict = {
        f"Ciudad_{ciudad_input}": 1,
        f"Vendedor_{vendedor_input}": 1,
        f"Nombre cliente_{cliente_input}": 1
    }
    input_vector = pd.DataFrame([input_dict])
    input_vector = input_vector.reindex(columns=X.columns, fill_value=0)

    predicted_total = model.predict(input_vector)[0]
    st.write(f"### Monto estimado: ${predicted_total:,.2f}")

    # Importancia de variables
    st.subheader("Importancia de variables")
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig_imp, ax_imp = plt.subplots(figsize=(10,6))
    sns.barplot(data=importance_df.head(20), x="Importance", y="Feature", ax=ax_imp)
    ax_imp.set_title("Top 20 variables más importantes")
    st.pyplot(fig_imp)
