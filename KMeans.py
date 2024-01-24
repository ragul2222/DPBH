# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline

# Load your dataset (replace 'your_dataset.csv' with your actual dataset file)
df = pd.read_csv('amazon_review.csv')

# Assume your dataset has a 'description' column containing product descriptions
corpus = df['description']

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(corpus)

# Dimensionality reduction using Truncated SVD
svd = TruncatedSVD(n_components=100, random_state=42)
X_svd = svd.fit_transform(X)

# Scale features for K-Means clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_svd)

# Determine the optimal number of clusters using silhouette score
best_score = -1
best_k = 2
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, labels)
    if silhouette_avg > best_score:
        best_score = silhouette_avg
        best_k = k

# Train K-Means with the optimal number of clusters
kmeans = KMeans(n_clusters=best_k, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to the dataset
df['cluster'] = labels

# Identify potential anomalies (outliers) as products in small clusters
anomaly_threshold = 0.05  # Adjust based on your dataset characteristics
anomalies = df[df['cluster'].value_counts(normalize=True) < anomaly_threshold]

# Print or further investigate the identified anomalies
print("Potential Anomalies:")
print(anomalies[['product_id', 'description', 'cluster']])

# Additional steps:
# - Incorporate other features (pricing, availability) into the model for a holistic analysis.
# - Evaluate the performance of the clustering algorithm and adjust parameters accordingly.
# - Integrate anomaly detection results with user reports and feedback for a comprehensive solution.