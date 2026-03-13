import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# Step 2: Load Dataset
# Make sure Mall_Customers.csv is in the same folder
df = pd.read_csv("Mall_Customers.csv")

# Show first few rows
print(df.head())

# Step 3: Select Features for Clustering
# We will use Annual Income and Spending Score
X = df.iloc[:, [3,4]].values

# Step 4: Find Optimal Number of Clusters using Elbow Method
sse = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure(figsize=(8,5))
plt.plot(range(1,11), sse, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("SSE (Sum of Squared Errors)")
plt.show()


# Step 5: Apply K-Means with Optimal Clusters (k=5)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)
from sklearn.metrics import silhouette_score

score = silhouette_score(X, y_kmeans)
print("Silhouette Score:", score)
print("Cluster Centers:")
print(kmeans.cluster_centers_)

# Step 6: Visualize the Clusters
plt.figure(figsize=(8,6))

plt.scatter(X[y_kmeans == 0,0], X[y_kmeans == 0,1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans == 1,0], X[y_kmeans == 1,1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans == 2,0], X[y_kmeans == 2,1], s=100, c='green', label='Cluster 3')
plt.scatter(X[y_kmeans == 3,0], X[y_kmeans == 3,1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[y_kmeans == 4,0], X[y_kmeans == 4,1], s=100, c='magenta', label='Cluster 5')

# Plot Centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
            s=300, c='yellow', label='Centroids')

plt.title("Customer Segmentation")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()

# Step 7: Revenue Prediction using Linear Regression

# Select features
X_rev = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Simulated revenue (based on spending score)
y_rev = df['Spending Score (1-100)'] * 10

# Create regression model
model = LinearRegression()

# Train the model
model.fit(X_rev, y_rev)

print("Revenue prediction model trained successfully")

# Predict revenue for a sample customer
income = 70
spending_score = 80

predicted_revenue = model.predict([[income, spending_score]])

print("Predicted Mall Revenue from this customer:", predicted_revenue[0])