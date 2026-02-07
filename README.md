# Simple XGBoost from Scratch with Fixed-Point Arithmetic

This project is a step-by-step, transparent reimplementation of the core ideas behind XGBoost with the long-term goal of enabling **zero-knowledge proofs (ZKPs)** for proving correct model training over some dataset.

We start with a minimal boosting framework for binary classification and progressively enhance it toward:

- Gradient and hessian boosting using logistic loss
- Fixed-point arithmetic
- Multiclass classification using vector gradients and softmax
- Benchmarking and evaluation against real XGBoost

---

## Features

- **SimpleXGBoostClassifier**: A from-scratch boosting model
- **FixedPointXGBoostClassifier**: Drop-in fixed-precision version using `decimal.Decimal` and Taylor approximations
- **Benchmark suite**: Easy-to-run comparison across real datasets vs real XGBoost
- **Stable numerics**: Clipped logits, stable sigmoid, Taylor-approximated softmax

---

## Motivation & Zero-Knowledge Proofs (ZKPs)

Our long-term goal is to use this implementation as the backend for **zero-knowledge proofs** that:

> "Prove that an XGBoost model was trained honestly with respect to a committed dataset, without revealing the dataset or model itself."

---

## Datasets & Benchmarks

Datasets used:
- Breast cancer
- Moons
- Circles
- Synthetic classification
- Digits (10 classes)
- Iris (3 classes)
- Wine (3 classes)
- Letter (26 classes)

Metrics computed:
- Accuracy
- F1 Score (weighted)

---

##  Development Journey

See [`xgboost_illustrated_guide.md`](./xgboost_illustrated_guide.md) for a narrative overview of how the project evolved, with diagrams and code insights.

---

## 📜 License
MIT License

