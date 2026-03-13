import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Select features
X = df.iloc[:, [3,4]].values

# Train K-Means model
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)

# Title
st.title("Mall Customer Segmentation")

st.write("Enter customer details to find the cluster.")

# User inputs
income = st.slider("Annual Income (k$)", 10,150)
score = st.slider("Spending Score (1-100)", 1,100)

# Predict cluster
prediction = kmeans.predict([[income,score]])

cluster_names = {
0: "Low Income - Low Spending",
1: "High Income - High Spending",
2: "Average Customers",
3: "Low Income - High Spending",
4: "High Income - Low Spending"
}

st.write("Customer belongs to Cluster:", prediction[0])
st.write("Customer Type:", cluster_names[prediction[0]])

# Show cluster graph
fig, ax = plt.subplots()

ax.scatter(X[:,0], X[:,1], c=kmeans.labels_)
ax.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='red', s=200)

ax.set_xlabel("Annual Income")
ax.set_ylabel("Spending Score")
ax.set_title("Customer Segments")

st.pyplot(fig)
