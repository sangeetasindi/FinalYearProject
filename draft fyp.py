import os
import re
import time
import random
import hashlib

import pandas as pd
import numpy as np
# This makes graphs show properly in Pycharm.
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
# Machine Learning tools
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from scipy.sparse import hstack, csr_matrix
from scipy.stats import ttest_rel


# SETTINGS
# This keeps results reproducible each time the code runs
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
# This shows the dataset path i have used
CSV_PATH = r"D:\fyp\enron_data_fraud_labeled.csv"
# Fast mode uses a smaller sample for quicker testing
# Quick run settings
FAST_MODE = True
FAST_SAMPLE_SIZE = 2500
SAMPLE_SIZE = 8000
# Maximum number of text features used by TF-IDF
TFIDF_MAX_FEATURES = 800
# Number of cross-validation folds
CV_SPLITS = 2
# Show plots at the end
SHOW_PLOTS = True
# Thresholds used for testing different alert levels
THRESHOLDS = [0.5, 0.6, 0.7]
# Folder for saved graphs
PLOT_DIR = "plots"


# KEYWORD LISTS
# Suspicious urgency-related words
URGENCY_WORDS = [
    "urgent", "immediately", "asap", "important", "action required",
    "now", "attention", "warning", "alert", "critical"
]
# Words linked to accounts and credentials
CREDENTIAL_WORDS = [
    "password", "login", "verify", "verification", "account",
    "reset", "confirm", "authentication", "security", "credential"
]
# Words linked to payments and money
MONEY_WORDS = [
    "invoice", "payment", "bank", "wire", "transfer",
    "money", "fund", "transaction", "refund", "credit"
]



# HELPERS
# This safely reads a numeric column from the dataset
# If the column is missing it will return a zero
def safe_numeric_column(df, col_name):
    if col_name in df.columns:
        return pd.to_numeric(df[col_name], errors="coerce").fillna(0)
    return pd.Series(0, index=df.index, dtype=float)

# This extracts the sender domain from the email address
def extract_domain(sender_text):
    if pd.isna(sender_text):
        return ""
    sender_text = str(sender_text).strip().lower()
    match = re.search(r'@([a-z0-9.-]+\.[a-z]{2,})', sender_text)
    return match.group(1) if match else ""

# This checks if the sender uses a common free email provider
def is_free_mail_domain(domain):
    free_domains = {
        "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
        "aol.com", "live.com", "msn.com"
    }
    return 1 if domain in free_domains else 0

# This measures how much of the email text is written in capitals
def uppercase_ratio(text):
    text = str(text)
    letters = sum(ch.isalpha() for ch in text)
    uppers = sum(ch.isupper() for ch in text)
    return uppers / letters if letters > 0 else 0.0

# This will count how many suspicous words appear in the email
def keyword_count(text, keywords):
    text = str(text).lower()
    total = 0
    for word in keywords:
        total += len(re.findall(rf"\b{re.escape(word)}\b", text))
    return total

# Hides a value by using hashing for privacy protection
def hash_value(value, salt="fyp_project"):
    if pd.isna(value):
        return ""
    return hashlib.sha256((salt + str(value)).encode()).hexdigest()

# Replaces sensitive information with tokens
# This keeps patterns but hides private details

def redact_sensitive_text(text):
    text = str(text)
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' EMAIL_TOKEN ', text)
    # Hides email address
    text = re.sub(r'https?://\S+|www\.\S+', ' URL_TOKEN ', text, flags=re.IGNORECASE)
    # Hides URLS
    text = re.sub(r'\b(?:\+?\d[\d\-\s]{7,}\d)\b', ' PHONE_TOKEN ', text)
    # Hides phone numbers
    text = re.sub(r'[$£€]\s?\d+(?:,\d{3})*(?:\.\d+)?', ' MONEY_TOKEN ', text)
    # Hides money amounts
    text = re.sub(r'\b\d{4,}\b', ' NUMBER_TOKEN ', text)
    # Hides long number sequence
    text = re.sub(r'\s+', ' ', text).strip()
    # Clean up spaces
    return text

# This prints basic dataset information
# This helps confirm the correct file was added
def show_dataset_info(df, csv_path):
    print("\n" + "=" * 70)
    print("DATASET USED")
    print("=" * 70)
    print("Path:", os.path.abspath(csv_path))
    print("Shape:", df.shape)
    print("Columns:")
    print(df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())

    if "Label" in df.columns:
        print("\nLabel counts:")
        print(df["Label"].value_counts(dropna=False))

    print("=" * 70)


# DATA LOADING
# This loads the dataset which checks the required columns
# Removes incomplete rows and samples of data if needed
def load_dataset(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    required_cols = ["Body", "Label"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.dropna(subset=["Body", "Label"]).copy()
    df["Body"] = df["Body"].astype(str)
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
    df = df.dropna(subset=["Label"])
    df["Label"] = df["Label"].astype(int)

    if FAST_MODE:
        n = min(FAST_SAMPLE_SIZE, len(df))
        df = df.sample(n=n, random_state=RANDOM_STATE)
    elif SAMPLE_SIZE:
        n = min(SAMPLE_SIZE, len(df))
        df = df.sample(n=n, random_state=RANDOM_STATE)

    return df


# COMMON FEATURE ENGINEERING
# Creates the main features used by both models
# These include sender features, body features,
# suspicious word counts, and structured dataset features
def build_common_features(df):
    df = df.copy()
# This gets sender information
    if "From" in df.columns:
        df["Sender_Raw"] = df["From"].fillna("").astype(str)
    else:
        df["Sender_Raw"] = ""
# Sender-based features
    df["Sender_Domain"] = df["Sender_Raw"].apply(extract_domain)
    df["Sender_Domain_Length"] = df["Sender_Domain"].str.len().astype(float)
    df["Sender_Is_FreeMail"] = df["Sender_Domain"].apply(is_free_mail_domain).astype(float)
    df["Sender_Missing"] = (df["Sender_Raw"].str.strip() == "").astype(float)
# Hidden sender version for privacy purposes
    df["Sender_Pseudonym"] = df["Sender_Raw"].apply(hash_value)
# Email body features
    df["Body_Length"] = df["Body"].str.len().astype(float)
    df["Num_URLs"] = df["Body"].str.count(r"https?://\S+|www\.\S+", flags=re.IGNORECASE).astype(float)
    df["Num_Emails"] = df["Body"].str.count(r"\b[\w\.-]+@[\w\.-]+\.\w+\b").astype(float)
    df["URL_Density"] = df["Num_URLs"] / (df["Body_Length"] + 1.0)
    df["Exclamations"] = df["Body"].str.count("!").astype(float)
# Suspicious behaviour indicators
    df["Uppercase_Ratio"] = df["Body"].apply(uppercase_ratio).astype(float)
    df["Urgency_Terms"] = df["Body"].apply(lambda x: keyword_count(x, URGENCY_WORDS)).astype(float)
    df["Credential_Terms"] = df["Body"].apply(lambda x: keyword_count(x, CREDENTIAL_WORDS)).astype(float)
    df["Money_Terms"] = df["Body"].apply(lambda x: keyword_count(x, MONEY_WORDS)).astype(float)
# Extra numeric features from the dataset
    df["Unique_Mails_From_Sender"] = safe_numeric_column(df, "Unique-Mails-From-Sender").astype(float)
    df["Suspicious_Folder"] = safe_numeric_column(df, "Suspicious-Folders").astype(float)
    df["Reply_Forward"] = safe_numeric_column(df, "Contains-Reply-Forwards").astype(float)

    return df



# PLOTTING
# This creates and saves the ROC curve and Precision Recall curve
def plot_metrics(y_test, probs, model_name):
    os.makedirs(PLOT_DIR, exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_")
# ROC curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_score = roc_auc_score(y_test, probs)

    fig1 = plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig1.savefig(os.path.join(PLOT_DIR, f"{safe_name}_roc.png"))
# Precision recall curve
    precision, recall, _ = precision_recall_curve(y_test, probs)
    baseline = y_test.mean()

    fig2 = plt.figure(figsize=(7, 5))
    plt.step(recall, precision, where="post", label="PR Curve")
    plt.axhline(y=baseline, linestyle="--", label=f"Baseline = {baseline:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig2.savefig(os.path.join(PLOT_DIR, f"{safe_name}_pr.png"))

    print(f"Plots saved for {model_name} in folder: {PLOT_DIR}")

    return [fig1, fig2]



# RUN ONE MODEL
# Runs one full experiment
# Prepare data
# Spilt data
# Convert text to numbers
# Combine text and numeric features
# Train the model
# Evaluate the model
def run_experiment(common_df, use_privacy):
    model_label = "PRIVACY-AWARE IDS" if use_privacy else "BASELINE EMAIL IDS"

    print("\n" + "=" * 70)
    print("MODEL:", model_label)
    print("=" * 70)

    df = common_df.copy()
# In privacy mode, hide senstive parts of the text first
    if use_privacy:
        df["Model_Text"] = df["Body"].apply(redact_sensitive_text)
    else:
        df["Model_Text"] = df["Body"]
# Spilt into training and testing sets
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["Label"],
        random_state=RANDOM_STATE
    )

    y_train = train_df["Label"].values
    y_test = test_df["Label"].values
# Convert email text into numbers using TF-IDF
    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        stop_words="english",
        min_df=2,
        dtype=np.float32
    )

    X_train_text = tfidf.fit_transform(train_df["Model_Text"])
    X_test_text = tfidf.transform(test_df["Model_Text"])
# Numeric features used alongside with text features
    numeric_cols = [
        "Sender_Domain_Length",
        "Sender_Is_FreeMail",
        "Sender_Missing",
        "Body_Length",
        "Num_URLs",
        "Num_Emails",
        "URL_Density",
        "Exclamations",
        "Uppercase_Ratio",
        "Urgency_Terms",
        "Credential_Terms",
        "Money_Terms",
        "Unique_Mails_From_Sender",
        "Suspicious_Folder",
        "Reply_Forward"
    ]
# Standardise numeric features
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(train_df[numeric_cols])
    X_test_num = scaler.transform(test_df[numeric_cols])
# Combine test and numeric features
    X_train = hstack([X_train_text, csr_matrix(X_train_num)])
    X_test = hstack([X_test_text, csr_matrix(X_test_num)])
# Lightweight classifier
    model = LogisticRegression(
        max_iter=200,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE
    )
# Cross validation check model stability
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    start_cv = time.time()
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )
    cv_time = time.time() - start_cv

    print(f"Mean CV ROC-AUC: {np.mean(cv_scores):.6f}")
    print(f"Cross-validation Time: {cv_time:.6f} seconds")
# Trains the final model
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train
# Predict the probability of each email being suspicious
    start_infer = time.time()
    probs = model.predict_proba(X_test)[:, 1]
    infer_time = time.time() - start_infer
# Main evaluation metrics
    roc_auc = roc_auc_score(y_test, probs)
    avg_precision = average_precision_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)

    print(f"Test ROC-AUC: {roc_auc:.6f}")
    print(f"Average Precision: {avg_precision:.6f}")
    print(f"Brier Score: {brier:.6f}")
    print(f"Training Time: {train_time:.6f} seconds")
    print(f"Inference Time: {infer_time:.6f} seconds")
# Create and save graphs
    figures = plot_metrics(y_test, probs, model_label)
# Test different thresholds for alert decisions
    print("\nThreshold-based intrusion alert analysis")
    for thr in THRESHOLDS:
        preds = (probs >= thr).astype(int)

        acc = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test,
            preds,
            average="binary",
            zero_division=0
        )
        cm = confusion_matrix(y_test, preds)

        print(f"\nThreshold: {thr}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1: {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)

    return cv_scores, figures



# MAIN
# Load dataset
# Show dataset information
# Build features
# Run baseline model
# Run privacy-aware mode
# Compare both models with a t test
if __name__ == "__main__":
    all_figures = []
# Load dataset
    base_df = load_dataset(CSV_PATH)
# Show dataset details
    show_dataset_info(base_df, CSV_PATH)
# Build common features once
    common_df = build_common_features(base_df)
# Run baseline model
    cv_non_priv, figs_non_priv = run_experiment(common_df, use_privacy=False)
    all_figures.extend(figs_non_priv)
# Run privacy aware model
    cv_priv, figs_priv = run_experiment(common_df, use_privacy=True)
    all_figures.extend(figs_priv)
# Compare both models statistically
    print("\n" + "=" * 70)
    print("STATISTICAL TEST (paired t-test)")
    print("=" * 70)

    t_stat, p_val = ttest_rel(cv_non_priv, cv_priv)

    print(f"t-statistic: {t_stat:.6f}")
    print(f"p-value: {p_val:.6f}")

    if p_val < 0.05:
        print("Significant difference detected.")
    else:
        print("No significant difference.")
# Shows the graphs at the end
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")
