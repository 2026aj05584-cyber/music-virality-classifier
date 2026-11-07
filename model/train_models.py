import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'viral_songs.csv')
    model_dir = os.path.join(base_dir, 'model')
    test_data_path = os.path.join(base_dir, 'test_data.csv')
    os.makedirs(model_dir, exist_ok=True)

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    # 12 predictive features
    categorical_features = ['genre']
    numerical_features = [
        'release_year',
        'duration_sec',
        'danceability',
        'energy',
        'valence',
        'tempo_bpm',
        'acousticness',
        'instrumentalness',
        'speechiness',
        'liveness',
        'loudness_db'
    ]
    feature_cols = categorical_features + numerical_features
    target_col = 'viral'

    X = df[feature_cols]
    y = df[target_col]

    # Stratified Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Save test dataset for app evaluation
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.to_csv(test_data_path, index=False)
    print(f"Saved test dataset (2000 rows, {len(test_df.columns)} columns) to: {test_data_path}")

    # Build Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ]
    )

    # Define 5 Classifiers
    models = {
        'logistic_regression': {
            'name': 'Logistic Regression',
            'clf': LogisticRegression(max_iter=1000, random_state=42)
        },
        'decision_tree': {
            'name': 'Decision Tree',
            'clf': DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=42)
        },
        'knn': {
            'name': 'KNN',
            'clf': KNeighborsClassifier(n_neighbors=7)
        },
        'naive_bayes': {
            'name': 'Naive Bayes',
            'clf': GaussianNB()
        },
        'support_vector_machine': {
            'name': 'Support Vector Machine',
            'clf': SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
        }
    }

    results = []

    print("\nTraining and evaluating models...\n")
    for key, item in models.items():
        name = item['name']
        clf = item['clf']

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        pipeline.fit(X_train, y_train)

        # Save trained pipeline model
        model_save_path = os.path.join(model_dir, f"{key}.joblib")
        joblib.dump(pipeline, model_save_path)
        print(f"[{name}] Saved model pipeline to: {model_save_path}")

        # Evaluate on test set
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        results.append({
            'key': key,
            'name': name,
            'Accuracy': round(acc, 4),
            'AUC': round(auc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1': round(f1, 4),
            'MCC': round(mcc, 4)
        })

    results_df = pd.DataFrame(results)
    print("\n--- Model Evaluation Comparison Table ---")
    print(results_df[['name', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']].to_string(index=False))

    # Save metrics JSON
    metrics_path = os.path.join(model_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics summary to: {metrics_path}")

if __name__ == '__main__':
    main()
