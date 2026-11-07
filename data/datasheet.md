# Datasheet: What Makes a Song Go Viral? — 10,000 Tracks with Audio Features

## Motivation

**What is this dataset for?**
A music dataset with Spotify-style audio features and a viral target. Designed for classification (predict viral hit), regression (predict streams), and audio feature analysis.

**Why was it created?**
Existing music datasets often lack a clear target variable or have licensing issues. This dataset provides a clean, fully-populated benchmark with realistic genre-specific audio distributions.

## Composition

**What do the instances represent?**
Each instance is a synthetic music track with audio features and streaming/viral metrics.

**How many instances?**
10,000 tracks spanning 2010–2025.

**What data does each instance consist of?**
- **Identifiers:** track_id, track_name, artist, genre, release_year
- **Audio features:** duration_sec, danceability, energy, valence, tempo_bpm, acousticness, instrumentalness, speechiness, liveness, loudness_db
- **Targets:** streams (regression), viral (classification, 1 if stream-based viral score exceeds threshold)

**Is there a label or target?**
Yes. `viral` is the binary classification target. `streams` is the regression target.

**Are there recommended data splits?**
Split by track_id. All rows are independent.

## Collection Process

**How was the data created?**
Synthetically generated via Python. Audio features follow genre-specific normal distributions based on published Spotify audio feature statistics. Viral status is derived from a weighted score of audio features plus genre tendency. Streams follow lognormal distributions with viral-boosted multipliers.

## Preprocessing

Data is generated in final format.

## Uses

- Binary classification: predict viral success
- Regression: predict streams from audio features
- Genre analysis: which genres over-index in virality?
- Feature importance: which audio features predict virality best?
- Teaching: class imbalance, feature engineering, log-transformed targets

**Should NOT be used for:**
- Real music industry decisions 
- Any commercial application involving real artists or labels

## Distribution

**License:** CC0 Public Domain

## Maintenance

**Maintainer:** Vedauv Rajkumar Leelavathi
**Contact:** https://www.kaggle.com/vedauv