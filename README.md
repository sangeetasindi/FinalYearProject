# Privacy-Aware Suspicious Email Detection Prototype
**Sangeeta Sindi**

**University of Greenwich**  
**Faculty of Engineering and Science**  
**School of Computing and Mathematical Sciences**  
**BSc (Hons) Computer Science**

This project implements a lightweight machine learning prototype for suspicious email detection. It compares a baseline model using original email text with a privacy-aware model using transformed text. The privacy-aware pipeline reduces exposure to sensitive information through sender-related hashing and text redaction while preserving useful patterns for classification.

---

## Overview

Email is widely used for digital communication, but it is also a common target for phishing, spam, and malicious activity. Traditional detection methods are effective for known threats, but they often struggle with new or evolving attack patterns. Many machine learning approaches also require direct access to detailed email content, which raises privacy concerns.

This project explores whether a lightweight privacy-aware suspicious email detection pipeline can preserve useful detection performance while reducing exposure to sensitive information.

Two experimental configurations are compared:

- **Baseline model** — uses original email text
- **Privacy-aware model** — uses transformed email text with sensitive content redacted

Both models use:

- **TF-IDF text features**
- **Metadata and behaviour-inspired engineered features**
- **Logistic Regression classifier**

---

## Main Features

- TF-IDF text representation
- Engineered metadata and behaviour-inspired features
- Logistic Regression classifier
- Sensitive text redaction
- Sender-related hashing
- ROC and Precision-Recall curve generation
- Threshold-based evaluation
- Cross-validation
- Paired t-test comparison

---

## Dataset

The project expects a CSV dataset containing at least the following columns:

### Required columns
- `Body`
- `Label`

### Optional columns
- `From`
- `Unique-Mails-From-Sender`
- `Suspicious-Folders`
- `Contains-Reply-Forwards`

If optional numeric columns are missing, the script automatically replaces them with zero values.

---

## Requirements

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib scikit-learn scipy





