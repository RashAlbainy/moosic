This bootcamp project explores unsupervised machine learning to cluster 5,000 Spotify songs into cohesive, human-like playlists for an imaginary music company called Moosic.

As Moosic’s catalog grows, playlists must still feel hand-curated — grouping tracks that sound right together while scaling automatically.

📊 Dataset

The dataset (from Spotify API) includes:

['name', 'artist', 'danceability', 'energy', 'key', 'loudness', 'mode',
 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
 'valence', 'tempo', 'duration_ms', 'time_signature', 'html']

⚙️ Methods & Workflow

Data Cleaning & Scaling

Dimensionality Reduction – PCA

Clustering Algorithms

K-Means

DBSCAN

HDBSCAN

Evaluation & Visualization

Silhouette Score

t-SNE Visualizations

🎧 Model Insights
K-Means

Works best with PCA-reduced data.

Dropping noisy features (time_signature, key) improves cluster quality.

Assigns every song to a cluster → no outliers.

Ideal for broad, mood-based playlists that mix genres and encourage discovery.

DBSCAN / HDBSCAN

Perform best on full feature space — all attributes contribute to fine-grained grouping.

Identify dense, well-defined clusters and mark sparse songs as outliers.

Perfect for genre-focused, consistent playlists.

HDBSCAN results:

27 clusters

Silhouette score: 0.211

Some large clusters (>400 songs) remain heterogeneous — suggesting manual curator review or semi-supervised refinement in future iterations.

🧩 Key Takeaways

K-Means → Broad, exploratory playlists (mood/energy-based).

HDBSCAN → Precise, genre-specific playlists (density-based).

Combine both:

Use unsupervised clustering as the backbone.

Add curator feedback for iterative, semi-supervised improvement.

🛠️ Tech Stack

Python

Pandas, NumPy

scikit-learn, HDBSCAN

Seaborn, Matplotlib

📈 Next Steps

Integrate listener feedback to refine clusters.

Experiment with audio embeddings or deep feature extraction.

Build a semi-supervised feedback loop for continuous playlist tuning.