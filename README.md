# Spotify Song Recommender

A content-based music recommendation system built in Python using the [Kaggle Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset). The user inputs a song name. Then the program recommends 5 similar songs using audio features and machine learning.

## How to run the project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

```bash
mkdir -p data/raw
kaggle datasets download -d maharshipandya/-spotify-tracks-dataset --unzip -p data/raw
```

### 3. Run the main script 
```bash
python main.py
```

***

## Example Output
```
Enter a song name or 'quit' to exit:
> Blinding Lights

Euclidean Method Recommendations
1. Save Your Tears by The Weeknd: pop
2. Levitating by Dua Lipa: pop
...
-------------------------------
Cosine Method Recommendations
1. Save Your Tears by The Weeknd: pop
2. As It Was by Harry Styles: pop
...
```

***

## How It Works

Recommendations are based on 12 audio features:

| Feature | Description |
|---|---|
| `danceability` | How suitable a track is for dancing (0–1) |
| `energy` | Perceptual measure of intensity and activity (0–1) |
| `valence` | Musical positiveness (0–1) |
| `tempo` | Estimated beats per minute |
| `acousticness` | Confidence the track is acoustic (0–1) |
| `instrumentalness` | Predicts whether a track contains no vocals (0–1) |
| `liveness` | Detects presence of a live audience (0–1) |
| `speechiness` | Presence of spoken words (0–1) |
| `loudness` | Overall loudness in decibels |
| `key` | Key the track is in |
| `mode` | Modality — major (1) or minor (0) |
| `time_signature` | Estimated time signature |

### Workflow

1. **Preprocessing** — Features are normalized with `MinMaxScaler` so no single feature dominates by magnitude (e.g. `tempo` vs `danceability`).
2. **KMeans Clustering** — Songs are grouped into 114 clusters (one per genre in the dataset) to narrow the search space before running KNN.
3. **KNN** — Within the input song's cluster, the 5 nearest neighbors are found using either Euclidean distance or Cosine similarity.

### Euclidean Distance
- Measures the straight line distance between two points in n-dimensional space 

### Cosine Similarity 
- Measures the angle between two vectors in n-dimensional space to see how much they point in the same direction 

## Visualizations

Exploratory data analysis is in `notebooks/visualize.ipynb` and includes:

- **Feature Comparison by Genre** — shows different graphs of certain features according to specific music genres
- **Audio Feature Correlation Heatmap** — shows covariance across the features of the dataset

***

## Dataset

- **Source:** [Kaggle — Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) by Maharshi Pandya
- **Size:** about 114,000 songs across 114 genres

***