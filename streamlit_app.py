import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.title("Streamlit Demo")

@st.cache_data
def load_data():
    return pd.read_csv("Iris.csv")

df = load_data()

st.subheader("Raw Data")
st.write(df.head())

df_clean = df.drop(columns=["Id", "Species"])

st.sidebar.header("Controls")
k = st.sidebar.slider("Number of Clusters (K)", 2, 6, 3)
x_axis = st.sidebar.selectbox("X Axis", df_clean.columns, index=0)
y_axis = st.sidebar.selectbox("Y Axis", df_clean.columns, index=2)

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(df_clean)

st.subheader("Cluster Visualization")

fig, ax = plt.subplots()
ax.scatter(df[x_axis], df[y_axis], c=df["Cluster"])
ax.set_xlabel(x_axis)
ax.set_ylabel(y_axis)
ax.set_title("K-Means Clusters")
st.pyplot(fig)

st.subheader("Cluster Centers")
centers = pd.DataFrame(kmeans.cluster_centers_, columns=df_clean.columns)
st.write(centers)

st.subheader("Cluster Summary")
st.write(df.groupby("Cluster")[df_clean.columns].mean())