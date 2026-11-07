# What Makes a Song Go Viral? — 10,000 Tracks with Audio Features

## What is this?

A synthetic music dataset with Spotify-style audio features and a clear viral target. 10,000 tracks, 2010–2025, single CSV.

## Why should you use it?

- **Clean, complete data** — no missing values, no messy joins
- **Defined target** — `viral` (1/0) for classification, `streams` for regression
- **Realistic audio distributions** — genre-specific feature ranges modeled on published Spotify statistics
- **Multiple tasks** — classification, regression, genre analysis
- **Universal appeal** — everyone has an opinion on what makes a hit

## Columns

| Column | Description |
|--------|-------------|
| `track_id` | Unique identifier |
| `track_name` | Synthetic track name |
| `artist` | Synthetic artist name |
| `genre` | Primary genre |
| `release_year` | Year released (2010–2025) |
| `duration_sec` | Track length in seconds |
| `danceability` | 0-1 score (how suitable for dancing) |
| `energy` | 0-1 score (intensity and activity) |
| `valence` | 0-1 score (musical positivity) |
| `tempo_bpm` | Beats per minute |
| `acousticness` | 0-1 score (confidence track is acoustic) |
| `instrumentalness` | 0-1 (predicts no vocals) |
| `speechiness` | 0-1 (presence of spoken words) |
| `liveness` | 0-1 (presence of audience) |
| `loudness_db` | Overall loudness in dB |
| `streams` | Total play count |
| `viral` | **Target:** 1 if viral, 0 otherwise |

## Suggested Tasks

- **Classification:** Predict `viral` from audio features
- **Regression:** Predict `streams` (log-transform first)
- **Analysis:** Which genres have the highest viral rate?
- **Feature importance:** Which audio features matter most?

## Quick Start

```python
import pandas as pd

df = pd.read_csv("viral_songs.csv")
print(df.shape)
print(f"Viral rate: {df['viral'].mean()*100:.1f}%")
print(df.head())
```

## License
CC0 Public Domain
