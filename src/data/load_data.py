from spotipy.oauth2 import SpotifyOAuth
import spotipy
import pandas as pd

"""
Extracts all songs from my public playlists on Spotify 

Note: All songs on Spotify is a significant amount of data. Therefore I will only extract songs from public playlists on my spotify profile 
"""


sp = spotipy.Spotify(
    auth_manager=SpotifyOAuth(
        client_id="CLIENT_ID", # Insert your Client ID here
        client_secret="CLIENT_SECRET", # Insert your Client secret here
        redirect_uri="REDIRECT_URI", # Insert your redirect uri here
    )
)


def get_songs(name, id):
    tracks = []

    # extract playlist metadata
    results = sp.playlist_items(id)

    while results:
        for item in results["items"]:
            track = item.get("item")
            if track and track["id"]:
                tracks.append(
                    {
                        "track_id": track["id"],
                        "track_name": track["name"],
                        "artist": track["artists"][0]["name"],
                        "artist_id": track["artists"][0]["id"],
                        "playlist": name,
                    }
                )

        results = sp.next(results)

    return tracks


def main():
    # insert name : id of each playlist (name can be anything)
    PLAYLISTS = {
        "10th grade": "6hbQUqSvyMJHuO1trHYAAD",
        "11th grade": "6N5LWmaTQoNHc1WknitIYZ",
        "2023 summer": "0c781bAyJOquTg57xKSyFb",
        "12th grade": "2jMEIY785NH6yZVWgnqDGT",
        "2024 summer": "5s479Piqg3jlrOUVre9aHC",
        "freshman": "7nB1XJFvmnSHuUQYodu9oi",
        "2025 summer": "1spItnBdAuvnAucEYrA0Vi",
        "sophomore s1": "0O7KhtmvhUMxilf4EelQHi",
        "sophomore s2": "4YPWjrQGORmLQ6GHefOTEj",
        "2026 summer": "0CI68T2EO5t8IIsVu5S6j5",
        "car": "4mYSo6xnb6HXjBhmvTJQ6v",
        "aa": "5Z71eKVJaZJNhNAgor7mYU",
    }

    all_tracks = []

    for name, id in PLAYLISTS.items():
        print(f'All songs from {name} playlist added')
        all_tracks.extend(get_songs(name, id))

    all_tracks = pd.DataFrame(all_tracks)
    all_tracks = all_tracks.drop_duplicates(
        subset=["track_name", "artist"], keep="first"
    ).reset_index()

    all_tracks.to_csv("../../data/raw/songs.csv")


if __name__ == "__main__":
    main()
