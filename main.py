#step 1: create a K means clustering graph where each cluster represents a genre
#step 2: group each song by their clusters
#step 3: narrow into each cluster with KNN and calculate both euclidean distance and cosine similarity to find which is optimal for song comparison

#use a graph representation
#weight = Euclidean distance or cosine similarity based on their audio features (danceability, energy, valence, etc.)
#use one big graph since it's possible to connect one genre to another