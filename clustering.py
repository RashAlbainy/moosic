import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
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
    from sklearn.cluster import DBSCAN
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import numpy as np
    import math
    from sklearn.manifold import TSNE
    from sklearn.neighbors import NearestNeighbors
    from sklearn.cluster import AgglomerativeClustering
    from scipy.cluster.hierarchy import dendrogram
    from sklearn.datasets import load_iris
    return (
        DBSCAN,
        KMeans,
        NearestNeighbors,
        PCA,
        StandardScaler,
        TSNE,
        math,
        np,
        pd,
        plt,
        silhouette_score,
        sns,
    )


@app.cell
def _(pd):
    df = pd.read_parquet("/Users/rasheedalbainy/Documents/Bootcamp/projects/moosic/Data/clean/spotify_final__clean_dataset.parquet", engine="fastparquet")
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
    return (df_scaled,)


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Clustering
    """)
    return


@app.cell
def _(PCA, df_scaled):

    features = [
        'danceability', 'energy', 'valence',
        'tempo', 'acousticness', 'instrumentalness','duration_ms'
    ]
    pca = PCA(n_components = 0.99)
    pca_data = pca.fit_transform(df_scaled[features])
    return (pca_data,)


@app.cell
def _(KMeans, df, df_scaled, pca_data):
    k = 25 # Choose number of clusters (start with a guess)
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42) # create model
    labels = kmeans.fit_predict(pca_data) # learn and apply
    df['cluster'] = labels # Add cluster labels to original df
    df_scaled['cluster'] = labels # Add cluster labels to  scaled df
    return kmeans, labels


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
def _(inertia_list, plt):


    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(inertia_list) + 1), inertia_list, marker='x')
    plt.title('Elbow Method showing the optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## silhouette
    """)
    return


@app.cell
def _(kmeans, pca_data, silhouette_score):
    silhouette_score(pca_data, kmeans.fit_predict(pca_data))
    return


@app.cell
def _(KMeans, pca_data, plt, silhouette_score):
    silhouette_scores = []
    cluster_range = range(2, 40)

    for z in cluster_range:
        kmeansS = KMeans(n_clusters=z, random_state=42)
        labelz = kmeansS.fit_predict(pca_data)
        score = silhouette_score(pca_data, labelz)
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
def _(df_scaled, math, np, plt):



    # 1️⃣ Select your plot_features (your chosen set)
    plot_features = [
        'danceability','energy','valence',
        'tempo','acousticness',
        'instrumentalness'
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
def _(TSNE, labels, pca_data, pd, plt, sns):


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # DBSCAN
    """)
    return


@app.cell
def _(df):
    df.columns
    return


@app.cell
def _(df_scaled):
    feat_df_scaled = df_scaled[[
        'danceability', 'energy', 'key', 'loudness',
           'speechiness', 'acousticness', 'instrumentalness', 'liveness',
           'valence', 'tempo', 'duration_ms'
    ]]

    return


@app.cell(hide_code=True)
def _(mo):
    epsilon = mo.ui.slider(
        label="Epsilon",
        start=0.3,    # ✅ instead of min
        stop=2.5,     # ✅ instead of max
        step=0.01,
        value=0.7,
    )

    min_samples = mo.ui.slider(
        label="Min_samples",
        start=3,
        stop=30,
        step=1,
        value=10,
    )

    mo.vstack([epsilon, min_samples])
    return epsilon, min_samples


@app.cell
def _(DBSCAN, df, df_scaled, epsilon, min_samples):
    db = DBSCAN(eps=epsilon.value, min_samples=min_samples.value).fit(df_scaled)
    db_labels = db.labels_
    df['db_clusters']=db_labels
    return db, db_labels


@app.cell(hide_code=True)
def _(db, df_scaled, silhouette_score):
    # Silhouette 
    db_sil_labels = db.fit_predict(df_scaled)
    mask = db_sil_labels != -1  # exclude noise points

    if len(set(db_sil_labels[mask])) > 1:  # must have ≥2 clusters to compute silhouette
        sil = silhouette_score(df_scaled[mask], db_sil_labels[mask])
        print(f"Silhouette score (no noise): {sil:.3f}")
    else:
        print("Not enough clusters to compute silhouette.")

    return


@app.cell(hide_code=True)
def _(TSNE, db_labels, df_scaled, pd, plt, sns):
    tsne_db = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_db_array = tsne_db.fit_transform(df_scaled)  # ✅ Works fine even if df_scaled is a NumPy array

    # Convert to DataFrame for plotting convenience
    tsne_db_results = pd.DataFrame(tsne_db_array, columns=['tsne0', 'tsne1'])

    # Add cluster labels
    tsne_db_results['db_cluster'] = pd.Categorical(db_labels)

    # Use a distinct color palette
    #nnn_clusters = tsne_db_results['db_cluster'].nunique()
    #palette_db = sns.color_palette("husl", nnn_clusters)
    nnn_clusters = tsne_db_results['db_cluster'].nunique()
    palette_db = (
       sns.color_palette("tab20") +
        sns.color_palette("tab20b") +
        sns.color_palette("tab20c") +
        sns.color_palette("Set3") +
        sns.color_palette("Paired")
    )[:nnn_clusters]

    # Plot the results
    sns.set(style="whitegrid")
    sns.scatterplot(
        data=tsne_db_results,
        x='tsne0',
        y='tsne1',
        hue='db_cluster',
        palette=palette_db,
        s=31,
        edgecolor="black",
        linewidth=0.3
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)   
    plt.title("t-SNE Visualization of dbscan Clusters")
    plt.show()
    return (tsne_db_results,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(NearestNeighbors, df_scaled, min_samples, sns):
    neigh = NearestNeighbors(n_neighbors=min_samples.value)
    neigh.fit(df_scaled)
    distances, indices = neigh.kneighbors(df_scaled)
    third_distances=distances[:,2]
    sorted_distances=sorted(third_distances)

    # Plot
    (
        sns.relplot(kind="line",
                    x=range(len(sorted_distances)),
                    y=sorted_distances)
        .set_axis_labels("Data Points sorted by distance",
                         "3rd Nearest Neighbour Distance (eps)")
        .set(title="k-distance Graph")
        .set(xlim=(1000, 5500))
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # HDBSCAN
    """)
    return


@app.cell
def _(df, df_scaled):
    import hdbscan

    hdb = hdbscan.HDBSCAN(min_cluster_size=35
                          , min_samples=3)
    hdb_labels = hdb.fit_predict(df_scaled)
    df['hdb_clusters']=hdb_labels
    return (hdb_labels,)


@app.cell
def _(df_scaled, hdb_labels, silhouette_score):
    hdb_mask = hdb_labels != -1
    if len(set(hdb_labels[hdb_mask])) > 1:
        hdb_sil = silhouette_score(df_scaled[hdb_mask], hdb_labels[hdb_mask])
        print(f"Silhouette (no noise): {hdb_sil:.3f}")
    else:
        print("Not enough clusters for silhouette.")
    return


@app.cell
def _(hdb_labels, plt, sns, tsne_db_results):
    sns.scatterplot(x=tsne_db_results['tsne0'],
                    y=tsne_db_results['tsne1'],
                    hue=hdb_labels,
                    palette=sns.color_palette("tab20") +
        sns.color_palette("tab20b") +
        sns.color_palette("tab20c") +
        sns.color_palette("Set3") +
        sns.color_palette("Paired"))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.title("HDBSCAN Clusters (t-SNE space)")
    plt.show()
    return


@app.cell
def _(df):
    df.groupby('hdb_clusters').count()
    return


@app.cell
def _(df):
    df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
