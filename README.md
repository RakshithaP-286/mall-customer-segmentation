# mall-customer-segmentation
1️⃣ Linear Regression

Definition

Linear Regression is a supervised machine learning algorithm used to predict continuous numerical values based on relationships between variables.

Formula

𝑦
=
𝑚
𝑥
+
𝑏
y=mx+b

Where:

y → predicted output

x → input variable

m → slope of line

b → intercept

In Your Project

Linear regression can be used to predict mall revenue based on customer income and spending score.

Example:

Input → Annual Income, Spending Score

Output → Estimated revenue

Key Points

Used for numerical prediction

Finds best-fit line

Minimizes prediction error

2️⃣ Logistic Regression

Definition

Logistic Regression is a supervised learning algorithm used for classification problems.

Instead of predicting numbers, it predicts probability of belonging to a class.

Formula

𝑃
=
1
1
+
𝑒
−
𝑧
P=
1+e
−z
1
	​


This is called the Sigmoid Function.

Example

Predict:

High spending customer

Low spending customer

Output is between:

0 and 1

Example:

Probability	Result
0.8	High spender
0.2	Low spender

Key Points

Used for binary classification

Uses sigmoid function

Output is probability

3️⃣ Clustering

Definition

Clustering is an unsupervised machine learning technique used to group similar data points together.

There are no predefined labels in clustering.

The algorithm finds hidden patterns in data.

Example in Your Project

Customers are grouped based on:

Annual Income
Spending Score

Output groups:

High income – high spending

High income – low spending

Low income – high spending

Low income – low spending

4️⃣ K-Means Clustering

Definition

K-Means is one of the most popular clustering algorithms.

It divides data into K number of clusters.

Working Steps

Choose number of clusters K

Select random centroids

Assign points to nearest centroid

Update centroid positions

Repeat until clusters stabilize

In Your Project

K = 5 clusters

Clusters represent different customer groups in the mall.

5️⃣ Elbow Method

Definition

The Elbow Method is used to find the optimal value of K for K-Means clustering.

It plots:

Number of clusters (K)
vs
SSE (Sum of Squared Errors)

The point where the graph bends like an elbow is chosen as the best K.

In your project:

Optimal K = 5
