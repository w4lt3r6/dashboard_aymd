import pandas as pd
from sklearn.cluster import KMeans
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
 
# Cargar datos
df = pd.read_excel("Base-2023-AYMD-OpenRefine (1).xlsx", engine="openpyxl")
df['Ciudad'] = df['Ciudad'].str.strip().str.lower()
client_sales = df.groupby(['Nombre cliente', 'Ciudad'])['Total'].sum().reset_index()
client_sales['Ciudad_code'] = client_sales['Ciudad'].astype('category').cat.codes
 
# Clustering
X = client_sales[['Ciudad_code', 'Total']]
kmeans = KMeans(n_clusters=4, random_state=42)
client_sales['Cluster'] = kmeans.fit_predict(X)
 
# Streamlit
st.title("Dashboard de Segmentación de Clientes AYMD")
st.write("Este dashboard muestra la segmentación de clientes por ciudad y monto total de compras usando K-Means Clustering.")
 
selected_cluster = st.selectbox("Selecciona un cluster:", sorted(client_sales['Cluster'].unique()))
filtered_data = client_sales[client_sales['Cluster'] == selected_cluster]
 
st.subheader("Clientes en el cluster seleccionado")
st.dataframe(filtered_data[['Nombre cliente', 'Ciudad', 'Total']])
 
st.subheader("Distribución de clientes por ciudad y monto total")
fig, ax = plt.subplots(figsize=(10,6))
sns.scatterplot(data=client_sales, x='Ciudad_code', y='Total', hue='Cluster', palette='Set2', ax=ax)
ax.set_title("Segmentación de clientes")
ax.set_xlabel("Ciudad (codificada)")
ax.set_ylabel("Total de compras")
st.pyplot(fig)
