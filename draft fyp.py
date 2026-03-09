import pandas as pd
import numpy as np
import hashlib
import re
import time
import random

# Machine learning tools
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
    brier_score_loss
)
from scipy.sparse import hstack, csr_matrix
from scipy.stats import ttest_rel


# =========================================================
# REPRODUCIBILITY SECTION
# =========================================================
# Setting fixed random seeds ensures the experiment
# produces identical results every time it runs.
# This is important for research validity.

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

CSV_PATH = r"D:\fyp\enron_data_fraud_labeled.csv"
SAMPLE_SIZE = 50000  # Using sample for computational efficiency
TFIDF_MAX_FEATURES = 5000  # Controls model size (lightweight goal)
THRESHOLDS = [0.5, 0.6, 0.7]  # Used to analyse fraud detection trade-offs


# =========================================================
# PRIVACY FUNCTION
# =========================================================
# This function applies SHA-256 hashing to sender identities.
# This simulates a privacy-preserving transformation.

def hash_value(value, salt="fyp_project"):
    if pd.isna(value):
        return ""
    return hashlib.sha256((salt + str(value)).encode()).hexdigest()


# =========================================================
# FEATURE ENGINEERING
# =========================================================
# This function prepares the dataset for modelling.
# It also switches between privacy and non-privacy modes.

def engineer_features(df, use_privacy):

    df = df.copy()
    df = df.dropna(subset=["Body", "Label"])

    # Ensure correct data types
    df["Body"] = df["Body"].astype(str)
    df["Label"] = df["Label"].astype(int)

    # Sampling to maintain lightweight computation
    if SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)

    # Privacy transformation
    if use_privacy:
        df["Sender_ID"] = df["From"].apply(hash_value)
    else:
        df["Sender_ID"] = df["From"].fillna("").astype(str)

    # Sender identity length as numerical signal
    df["Sender_Length"] = df["Sender_ID"].apply(len).astype(float)

    # Behavioural fraud indicators
    url_pattern = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

    df["Body_Length"] = df["Body"].apply(len)
    df["Num_URLs"] = df["Body"].apply(lambda x: len(url_pattern.findall(x)))
    df["URL_Density"] = df["Num_URLs"] / (df["Body_Length"] + 1)
    df["Exclamations"] = df["Body"].apply(lambda x: x.count("!"))

    # Convert metadata features safely
    df["Unique-Mails-From-Sender"] = pd.to_numeric(
        df.get("Unique-Mails-From-Sender", 0), errors="coerce"
    ).fillna(0)

    df["Suspicious_Folder"] = pd.to_numeric(
        df.get("Suspicious-Folders", 0), errors="coerce"
    ).fillna(0)

    df["Reply_Forward"] = pd.to_numeric(
        df.get("Contains-Reply-Forwards", 0), errors="coerce"
    ).fillna(0)

    return df


# =========================================================
# EXPERIMENT PIPELINE
# =========================================================
# This function runs the full experiment for either:
# - Raw sender identity (non-privacy)
# - Hashed sender identity (privacy)

def run_experiment(use_privacy):

    print("\n" + "=" * 70)
    print("MODEL:", "PRIVACY (hashed sender)" if use_privacy else "NON-PRIVACY (raw sender)")
    print("=" * 70)

    # Load dataset
    df = pd.read_csv(CSV_PATH, low_memory=False)

    # Apply feature engineering
    df = engineer_features(df, use_privacy)

    # Stratified split preserves fraud ratio
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["Label"],
        random_state=RANDOM_STATE
    )

    y_train = train_df["Label"].values
    y_test = test_df["Label"].values

    # =========================================================
    # TEXT FEATURES (TF-IDF)
    # =========================================================
    # Converts email text into numerical vectors.
    # Limited to 5000 features to maintain lightweight design.

    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        stop_words="english"
    )

    X_train_text = tfidf.fit_transform(train_df["Body"])
    X_test_text = tfidf.transform(test_df["Body"])

    # =========================================================
    # NUMERIC FEATURES
    # =========================================================

    numeric_cols = [
        "Sender_Length",
        "Body_Length",
        "Num_URLs",
        "URL_Density",
        "Exclamations",
        "Unique-Mails-From-Sender",
        "Suspicious_Folder",
        "Reply_Forward"
    ]

    scaler = StandardScaler()

    X_train_num = scaler.fit_transform(train_df[numeric_cols])
    X_test_num = scaler.transform(test_df[numeric_cols])

    # Combine sparse TF-IDF with numeric signals
    X_train = hstack([X_train_text, csr_matrix(X_train_num)])
    X_test = hstack([X_test_text, csr_matrix(X_test_num)])

    # =========================================================
    # MODEL SELECTION
    # =========================================================
    # Logistic Regression chosen because:
    # - Computationally lightweight
    # - Interpretable
    # - Suitable for high-dimensional sparse data

    model = LogisticRegression(
        max_iter=1500,
        class_weight="balanced",  # Handles class imbalance
        solver="lbfgs"
    )

    # Cross-validation for robustness
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

    print("Mean CV ROC-AUC:", np.mean(cv_scores))

    # Training time measurement
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    # Inference time measurement
    start_inf = time.time()
    probs = model.predict_proba(X_test)[:, 1]
    infer_time = time.time() - start_inf

    # Primary evaluation metrics
    roc_auc = roc_auc_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)

    print("Test ROC-AUC:", roc_auc)
    print("Brier Score:", brier)
    print("Training Time:", train_time)
    print("Inference Time:", infer_time)
    print("Number of Parameters:", model.coef_.shape[1])

    # =========================================================
    # THRESHOLD ANALYSIS
    # =========================================================
    # Demonstrates trade-off between precision and recall.

    for thr in THRESHOLDS:
        preds = (probs >= thr).astype(int)

        acc = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="binary", zero_division=0
        )

        print(f"\nThreshold: {thr}")
        print("Accuracy:", acc)
        print("Precision (fraud):", prec)
        print("Recall (fraud):", rec)
        print("F1 (fraud):", f1)
        print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

    return cv_scores


# =========================================================
# MAIN EXECUTION
# =========================================================

if __name__ == "__main__":

    # Run both configurations
    cv_non_priv = run_experiment(use_privacy=False)
    cv_priv = run_experiment(use_privacy=True)

    # Statistical significance test
    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISON (Paired t-test)")
    print("=" * 70)

    t_stat, p_val = ttest_rel(cv_non_priv, cv_priv)

    print("t-statistic:", t_stat)
    print("p-value:", p_val)

    if p_val < 0.05:
        print("Statistically significant difference detected.")
    else:
        print("No statistically significant difference detected.")