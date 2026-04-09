import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Title 
st.title("Streamlit Demo")

# Cache
@st.cache_data
def load_data():
    return pd.read_csv("Iris.csv")

# Load the dataset
df = load_data()

# Subheader
st.subheader("Raw Data")
st.write(df.head())

# Checkbox for full data 
show_data = st.checkbox("Show full dataset")

if show_data:
    st.write(df)


# Clean data
df_clean = df.drop(columns=["Id", "Species"])

# Sidebar
st.sidebar.header("Controls")

# Slider to choose number of clusters 
k = st.sidebar.slider("Number of Clusters (K)", 2, 6, 3)

# Dropdowns to pick which features to plot
x_axis = st.sidebar.selectbox("X Axis", df_clean.columns, index=0)
y_axis = st.sidebar.selectbox("Y Axis", df_clean.columns, index=2)

# Create KMeans model
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

# Fit model 
df["Cluster"] = kmeans.fit_predict(df_clean)

# Plot the clusters
fig, ax = plt.subplots()

# Scatter plot of selected features
scatter = ax.scatter(df[x_axis], df[y_axis], c=df["Cluster"])

# Label axes
ax.set_xlabel(x_axis)
ax.set_ylabel(y_axis)
ax.set_title("K-Means Clustering")

# Show plot in Streamlit
st.pyplot(fig)