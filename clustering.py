import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # scaling
    """)
    return


@app.cell
def _():
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    return KMeans, StandardScaler, pd, silhouette_score


@app.cell
def _(pd):
    df = pd.read_parquet("/Users/rasheedalbainy/Documents/Bootcamp/projects/moosic/Data/clean/spotify_final_dataset.parquet", engine="fastparquet")
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(StandardScaler, df, pd):
    X = df.select_dtypes(include='number') # includes only numeric columns
    scaler = StandardScaler() # create the model
    X_scaled = scaler.fit_transform(X) # learn from the data and apply to the assigned data
    df_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=df.index) # show me these results in my dataframe
    return X, df_scaled


@app.cell
def _(df):
    df
    return


@app.cell
def _(df_scaled):
    df_scaled
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Clustering
    """)
    return


@app.cell
def _(df_scaled):
    from sklearn.decomposition import PCA
    features = [
        'danceability', 'energy', 'valence',
        'tempo', 'speechiness', 'acousticness', 'instrumentalness', 'duration_ms', 'liveness'
    ]
    pca = PCA(n_components = 0.9)
    pca_data = pca.fit_transform(df_scaled[features])
    return (pca_data,)


@app.cell
def _(KMeans, df, df_scaled, pca_data):
    k = 31 # Choose number of clusters (start with a guess)
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42) # create model
    labels = kmeans.fit_predict(pca_data) # learn and apply
    df['cluster'] = labels # Add cluster labels to original df
    df_scaled['cluster'] = labels # Add cluster labels to  scaled df
    return kmeans, labels


@app.cell
def _(mo):
    mo.md(r"""
    features = [
      'danceability', 'energy', 'valence',
      'tempo', 'speechiness', 'acousticness', 'instrumentalness', 'duration_ms', 'liveness'
    ]
    k = 25 # Choose number of clusters (start with a guess)
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42) # create model
    labels = kmeans.fit_predict(df_scaled[features]) # learn and apply
    df['cluster'] = labels # Add cluster labels to original df
    df_scaled['cluster'] = labels # Add cluster labels to  scaled df
    """)
    return


@app.cell
def _(df_scaled):
    df_scaled
    return


@app.cell
def _(df_scaled):
    df_scaled.groupby('cluster')[['speechiness', 'instrumentalness']].median().plot(kind='bar', figsize=(10,6))
    return


@app.cell
def _(df_scaled):
    df_scaled.groupby('cluster')[['speechiness', 'instrumentalness']].mean().plot(kind='bar', figsize=(10,6))
    return


@app.cell
def _(kmeans):
    kmeans.inertia_
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## inertia
    """)
    return


@app.cell
def _(KMeans, df_scaled):
    inertia_list = []

    for i in range(1,40):
        myKMeans = KMeans(n_clusters=i)
        myKMeans.fit(df_scaled)
        inertia_list.append(round(myKMeans.inertia_))
    return (inertia_list,)


@app.cell
def _(inertia_list):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(inertia_list) + 1), inertia_list, marker='x')
    plt.title('Elbow Method showing the optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()
    return (plt,)


@app.cell
def _(mo):
    mo.md(r"""
    # evaluating
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## silhouette
    """)
    return


@app.cell
def _(X, df_scaled, kmeans, silhouette_score):
    silhouette_score(df_scaled, kmeans.fit_predict(X))
    return


@app.cell
def _(KMeans, df_scaled, plt, silhouette_score):
    silhouette_scores = []
    cluster_range = range(10, 50)  # Usually start from 2 because silhouette is undefined for 1 cluster

    for z in cluster_range:
        kmeansS = KMeans(n_clusters=z, random_state=42)
        labelz = kmeansS.fit_predict(df_scaled)
        score = silhouette_score(df_scaled, labelz)
        silhouette_scores.append(score)

    plt.plot(cluster_range, silhouette_scores, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Different Values of k')

    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Radar chart
    """)
    return


@app.cell
def _(df_scaled, plt):
    import numpy as np
    import math


    # 1️⃣ Select your plot_features (your chosen set)
    plot_features = [
        'danceability','energy','valence',
        'tempo','speechiness','acousticness',
        'instrumentalness','duration_ms','liveness'
    ]
    plot_features = [f for f in plot_features if f in df_scaled.columns]

    # 2️⃣ Compute cluster means on scaled data
    cluster_means = df_scaled.groupby('cluster')[plot_features].mean().sort_index()

    # 3️⃣ Radar setup
    n = len(plot_features)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    angles = np.r_[angles, angles[0]]  # close loop

    # 4️⃣ Create subplots for each cluster
    n_clusters = cluster_means.shape[0]
    cols = 5
    rows = math.ceil(n_clusters / cols)

    fig, axes = plt.subplots(rows, cols, subplot_kw=dict(polar=True), figsize=(3.8*cols, 3.4*rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (cl, row) in zip(axes, cluster_means.iterrows()):
        vals = np.r_[row.values, row.values[0]]  # close loop
        ax.plot(angles, vals, linewidth=1.2)
        ax.fill(angles, vals, alpha=0.15)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(plot_features, fontsize=8)
        ax.set_yticklabels([])
        ax.set_title(f'Cluster {cl}', y=1.06, fontsize=10)

    # Hide unused subplots
    for ax in axes[n_clusters:]:
        ax.axis('off')

    fig.suptitle('Cluster Profiles (scaled plot_features)', y=1.02)
    fig.tight_layout()
    plt.show()

    return


@app.cell
def _(mo):
    mo.md(r"""
    ## tsne
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(labels, pca_data, pd, plt):
    from sklearn.manifold import TSNE
    import seaborn as sns
 
    # t-SNE takes arrays (pca_data is a NumPy array)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_array = tsne.fit_transform(pca_data)  # ✅ Works fine even if pca_data is a NumPy array

    # Convert to DataFrame for plotting convenience
    tsne_results = pd.DataFrame(tsne_array, columns=['tsne0', 'tsne1'])

    # Add cluster labels
    tsne_results['Cluster'] = pd.Categorical(labels)

    # Use a distinct color palette
    nn_clusters = tsne_results['Cluster'].nunique()
    palette = (
        sns.color_palette("tab20") +
        sns.color_palette("tab20b") +
        sns.color_palette("tab20c") +
        sns.color_palette("Set3") +
        sns.color_palette("Paired")
    )[:nn_clusters]

    # Plot the results
    sns.set(style="whitegrid")
    sns.scatterplot(
        data=tsne_results,
        x='tsne0',
        y='tsne1',
        hue='Cluster',
        palette=palette,
        s=31,
        edgecolor="black",
        linewidth=0.3
    )

    plt.title("t-SNE Visualization of PCA + KMeans Clusters")
    plt.show()
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
