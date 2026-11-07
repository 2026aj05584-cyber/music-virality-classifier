# Music Virality Prediction & Classification

**Course:** M.Tech (AIML) — Machine Learning  
**Institution:** BITS Pilani (Work Integrated Learning Programmes Division)  
**Assignment:** Assignment - 2  

---

## a. Problem Statement

In the digital music and streaming era, record labels, streaming platforms, and independent artists release thousands of tracks daily. Identifying which tracks have the acoustic profile and metadata characteristics necessary to become a viral hit is a high-value challenge.

The goal of this project is to develop and deploy an end-to-end Machine Learning classification pipeline that predicts whether a music track will go **viral** (`viral = 1`) or not (`viral = 0`) based on its genre, release year, and quantitative Spotify-style audio characteristics. The project includes training, tuning, and evaluating five diverse supervised classification models, packaging the trained pipelines, and deploying an interactive dashboard using Streamlit for real-time model evaluation on unseen test datasets.

---

## b. Dataset Description

* **Dataset Name:** What Makes a Song Go Viral? — 10,000 Tracks with Audio Features
* **Source:** [Kaggle - What Makes a Song Go Viral? (10,000 Tracks)](https://www.kaggle.com/datasets/vedauv/what-makes-a-song-go-viral-10000-tracks)
* **Total Instances:** 10,000 tracks
* **Total Features Used:** 12 predictive features (1 categorical + 11 numerical)
* **Target Variable:** `viral` (Binary: `1` for viral hit, `0` for non-viral)
* **Class Distribution:** Non-Viral (`0`): 63.95% | Viral (`1`): 36.05%
* **Data Split:** 80% Training (8,000 samples) and 20% Holdout Test (2,000 samples) using stratified sampling.

### Feature Details:
1. `genre` (Categorical): Musical genre (`Pop`, `Electronic`, `Hip-Hop`, `Classical`, `Jazz`, `Latin`, `Rock`, `Indie`, `R&B`, `Country`).
2. `release_year` (Numerical): Year the track was released (2010–2025).
3. `duration_sec` (Numerical): Duration of the track in seconds.
4. `danceability` (Numerical): Suitability for dancing (0.0 to 1.0).
5. `energy` (Numerical): Intensity and active perceptual measure (0.0 to 1.0).
6. `valence` (Numerical): Musical positiveness conveyed by the track (0.0 to 1.0).
7. `tempo_bpm` (Numerical): Estimated tempo in beats per minute.
8. `acousticness` (Numerical): Confidence measure of whether the track is acoustic (0.0 to 1.0).
9. `instrumentalness` (Numerical): Predicts whether a track contains no vocals (0.0 to 1.0).
10. `speechiness` (Numerical): Detects the presence of spoken words (0.0 to 1.0).
11. `liveness` (Numerical): Detects presence of an audience/live setting (0.0 to 1.0).
12. `loudness_db` (Numerical): Overall loudness of the track in decibels (dB).

*(Identifiers such as `track_id`, `track_name`, `artist`, and the continuous regression target `streams` were excluded from training to prevent data leakage).*

---

## c. Github Repository Link

* **GitHub Repository:** [https://github.com/2026aj05584-cyber/music-virality-classifier](https://github.com/2026aj05584-cyber/music-virality-classifier)
* **Live Streamlit App:** [https://music-virality-classifier-aftzyu76sekuncn36m6nbj.streamlit.app/](https://music-virality-classifier-aftzyu76sekuncn36m6nbj.streamlit.app/)

---

## d. Models Used & Comparison Table

Five supervised machine learning classification algorithms were implemented and trained on the same preprocessed training set. Features were normalized using `StandardScaler` for continuous numerical features and `OneHotEncoder` for the categorical `genre` feature within an isolated `Pipeline` to eliminate data leakage.

### Model Evaluation Comparison Table (on 2,000 Test Samples):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.7360 | 0.8081 | 0.6421 | 0.6047 | 0.6229 | 0.4205 |
| **Decision Tree** | 0.7035 | 0.7653 | 0.5930 | 0.5659 | 0.5791 | 0.3507 |
| **KNN** | 0.6885 | 0.7472 | 0.5714 | 0.5437 | 0.5572 | 0.3174 |
| **Naive Bayes** | 0.6935 | 0.8062 | 0.5464 | 0.8821 | 0.6748 | 0.4568 |
| **Support Vector Machine** | **0.7380** | 0.7808 | **0.6468** | 0.6019 | 0.6236 | 0.4237 |

---

## Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved the highest **AUC score (0.8081)** and strong accuracy (73.60%). Because acoustic attributes like energy, danceability, and loudness often exhibit steady monotonic relationships with audience engagement, the linear decision boundary generalizes cleanly without overfitting. |
| **Decision Tree** | Obtained 70.35% accuracy and an AUC of 0.7653. The tree model captured distinct orthogonal threshold cuts (e.g. genre splits and danceability cutoffs), but its axis-aligned partitioning led to higher variance and slightly lower generalization on continuous metrics. |
| **KNN** | Yielded the lowest accuracy (68.85%) and AUC (0.7472). In a 12-dimensional feature space, distance metrics suffer from the curse of dimensionality, making nearest-neighbor clustering less distinct across overlapping acoustic profiles. |
| **Naive Bayes** | Demonstrated exceptional **Recall (88.21%)**, highest **F1 (0.6748)**, and highest **MCC (0.4568)**. While the strong feature independence assumption slightly penalizes Precision (0.5464) due to correlated features (e.g. loudness and energy), it serves as an outstanding high-sensitivity screening tool that rarely misses a viral hit. |
| **Support Vector Machine** | Achieved the highest **Accuracy (73.80%)** and highest **Precision (64.68%)**. The radial basis function (RBF) kernel effectively captured complex non-linear acoustic interactions in the normalized feature space, maximizing margin separation between viral and non-viral songs. |

### Overall, Winner for your dataset?

* **Winner for Overall Accuracy and Precision:** **Support Vector Machine (SVM)** (Accuracy: **73.80%**, Precision: **64.68%**).
* **Winner for Ranking Discrimination and Probability Calibration:** **Logistic Regression** (AUC: **0.8081**).
* **Overall Recommended Production Model:** **Support Vector Machine (SVM)** is selected as the overall winner because it achieves the highest predictive accuracy and minimizes false positives (highest precision), which is crucial for industry stakeholders investing budget into prospective viral tracks.

---

## Repository Structure

```text
Assignment2/
│-- app.py                     # Interactive Streamlit Web Application
│-- requirements.txt           # Python dependency requirements for deployment
│-- README.md                  # Project documentation, metrics table & analysis
│-- test_data.csv              # 20% holdout test dataset (2,000 rows with 12 features + viral)
│-- model/
│   │-- train_models.py        # Model training and pipeline generation script
│   │-- metrics.json           # Serialized metric evaluation results
│   │-- logistic_regression.joblib
│   │-- decision_tree.joblib
│   │-- knn.joblib
│   │-- naive_bayes.joblib
│   │-- support_vector_machine.joblib
│-- data/
│   │-- viral_songs.csv        # Full dataset (10,000 tracks)
│   │-- datasheet.md           # Dataset documentation
```

---

## How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/2026aj05584-cyber/music-virality-classifier.git
   cd music-virality-classifier
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit App:**
   ```bash
   streamlit run app.py
   ```
