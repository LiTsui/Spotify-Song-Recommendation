from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler  # necessary to allow each feature to have equal weight
import pandas as pd
import numpy as np

"""
A simple program that recommends songs 
"""

np.seterr(divide="ignore", invalid="ignore", over="ignore")  # necessary to ignore errors

df = pd.read_csv("data/dataset.csv")
df = df.dropna().drop_duplicates(["track_name"]).reset_index(drop=True)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(
    df.drop(
        columns=[
            "track_id",
            "artists",
            "album_name",
            "track_name",
            "explicit",
            "track_genre",
        ]
    )
)


def recommend(song, metric):
    row = df[df["track_name"].str.lower() == song.lower()]

    k_means = KMeans(n_clusters=114, random_state=42)  # num of song genres
    df["cluster"] = k_means.fit_predict(scaled)  # assign each row to a cluster

    knn_models = {}
    for cluster_id in df["cluster"].unique():  # iterate each cluster
        cluster_indices = df[df["cluster"] == cluster_id].index
        cluster_scaled = scaled[
            cluster_indices
        ]  # gets all scaled information for that specific cluster
        knn = NearestNeighbors(n_neighbors=6, metric=metric) # get 6 closest points
        knn.fit(cluster_scaled)
        knn_models[cluster_id] = (knn, cluster_indices)

    idx = row.index[0] # first song recommended is the input song itself
    cluster_id = df.loc[idx, "cluster"]
    knn, cluster_indices = knn_models[cluster_id]
    distances, local_indices = knn.kneighbors([scaled[idx]])
    global_indices = cluster_indices[local_indices[0][1:]]
    format(df[["track_name", "artists", "track_genre"]].iloc[global_indices])


def format(df):
    df = df.reset_index(drop=True)
    for row in df.itertuples():
        print(f"{row.Index + 1}. {row.track_name} by {row.artists}: {row.track_genre}")


def main():
    while True:
        song = input("Enter a song name or 'quit' to exit:\n")
        if song.lower() == "quit":
            break

        row = df[df["track_name"].str.lower() == song.lower()]
        if row.empty:
            print(
                f"Song '{song}' does not exist in the dataset. Make sure to double check your song name."
            )
        else:
            print("Euclidean Method Recommendations")
            recommend(song, "euclidean")
            print("-------------------------------")
            print("Cosine Method Recommendations")
            recommend(song, "cosine")
            print("-------------------------------")


if __name__ == "__main__":
    main()
