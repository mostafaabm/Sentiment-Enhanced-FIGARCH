# Forecasting Asset Returns with Sentiment-Enhanced FIGARCH Models

This repository contains the codebase and datasets for the paper: **"Forecasting Asset Returns with Sentiment-Enhanced FIGARCH Models."** 

This project introduces a novel framework that integrates machine-learning-derived news impact probabilities (via Random Forest) and FinBERT textual sentiment scores into Fractionally Integrated GARCH (FIGARCH) models. The objective is to forecast asset returns and model volatility dynamics across multiple distinct intraday time frames (Open-to-Close, Open-to-Open, Close-to-Close, and Close-to-Open).

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Dependencies](#dependencies)
- [How to Run the Code](#how-to-run-the-code)
- [Contact](#contact)

---

## Overview
This study investigates how sentiment effects differ across specific intraday windows. We utilize a dataset of over 347,000 asset-specific news headlines covering 37 assets across the stock, cryptocurrency, and commodity markets (March 2022 to January 2025).

The code provided here reproduces the two primary stages of the research:
1. **Natural Language Processing & Machine Learning:** Generating Bag-of-Words (TF-IDF) features and training Random Forest classifiers to compute time-frame-specific impact probabilities.
2. **Econometric Volatility Modeling:** Fitting FIGARCH(1,1) models augmented with these exogenous machine-learning and sentiment variables to evaluate their contribution to model fit (AIC, BIC, LLF) and statistical significance.

---

## Repository Structure

To ensure the scripts run out-of-the-box, the repository is structured as follows:

```text
Sentiment-Enhanced-FIGARCH/
│
├── README.md                           <- Project overview and instructions
├── AlphaVantage_BOW_dataset.py         <- Script 1: NLP, TF-IDF, and Random Forest classification
├── Alpha_BOW_FIGARCH.py                <- Script 2: FIGARCH modeling and statistical evaluation
│
├── data/
│   ├── Alpha_FinBERT_datasets/         <- Raw news headlines and FinBERT sentiment scores
│   └── Frequency-based_Datasets/       <- Target variables (log returns) and asset data
│
└── results/                            <- Output folder for Excel summary reports (auto-generated)
```
*(Note: Datasets are provided for reproducibility. Due to GitHub file size limits, some larger datasets may be provided as samples).*

---

## Dependencies
The code is written in **Python 3.x**. To run the scripts, you will need the following libraries installed. You can install them via `pip`:

```bash
pip install pandas numpy matplotlib scikit-learn nltk spacy arch xlsxwriter wordcloud transformers
```

Additionally, you must download the specific English language model for `spaCy` to perform NLP tokenization and lemmatization:
```bash
python -m spacy download en_core_web_sm
```

---

## How to Run the Code

The analysis is divided into two sequential scripts. Run them in the following order from the root directory of this repository:

### 1. Generate Impact Probabilities
Run the machine-learning pipeline to preprocess the text and train the Random Forest classifiers:
```bash
python AlphaVantage_BOW_dataset.py
```
* **What it does:** Reads the raw datasets, extracts TF-IDF features, applies custom financial stopwords/synonyms, tunes the Random Forest via GridSearch (strictly on training data to prevent leakage), and outputs the impact probabilities for positive/negative directional movements across all time frames.
* **Output:** Saves `RF_BOW_summary.xlsx` and generated datasets into the `./data/` and `./results/` directories.

### 2. Fit the FIGARCH Models
Run the econometric pipeline to evaluate the exogenous variables:
```bash
python Alpha_BOW_FIGARCH.py
```
* **What it does:** Iterates through 8 distinct sets of exogenous variables across 4 time horizons. It scales the inputs, fits the FIGARCH(1,1) models, and calculates the statistical significance (p-values) of all core and exogenous parameters while avoiding look-ahead bias in the data alignment.
* **Output:** Saves `FIGARCH_Comparative_Results.xlsx` to the `./results/` directory, detailing AIC, BIC, LLF, and variable significance counts.
