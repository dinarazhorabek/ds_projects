# Inertia Trading – Boeing (BA) & S&P 500 (SPY)
### Machine Learning, Rule-Based Strategies, and Clustering Analysis

## Overview

This repository explores short-term trading behavior in Boeing (BA) using a combination of rule-based trading, supervised machine learning, and unsupervised clustering.  
SPY (S&P 500 ETF) is included as a baseline benchmark in the initial analysis.

The project consists of three main components:

1. **Day Trading – Inertia Strategy (BA & SPY)**  
   A rule-based approach that uses daily open–close dynamics to test whether overnight price movements persist into the next trading day.  
   Each position opens at the market open and closes at the same day’s close, using a fixed $100 per trade.

2. **Weekly Trading – Machine Learning Labeling Strategy (BA)**  
   A classification-based approach using weekly features such as mean return and volatility.  
   Multiple ML models—including Logistic Regression, kNN, SVMs, Decision Trees, Random Forest, Naïve Bayes, and LDA/QDA—are trained on **2023–2024** data and tested out-of-sample on **2020–2022** to label weeks as **green (buy)** or **red (cash)** and evaluate simulated trading performance.

3. **Clustering Analysis (BA + Dow Jones Stocks)**  
   Weekly (μ, σ) patterns are analyzed using k-means clustering to identify natural behavioral regimes in BA.  
   Additional clustering across five Dow Jones stocks (AMZN, JNJ, MCD, NKE, NVDA) compares cross-stock behavior using cluster trajectories and Hamming distances to measure stability and similarity.


---

## 1. Day Trading – Inertia Strategy

### Concept
The inertia approach assumes that overnight price changes often persist during the following trading day.  
Using daily `Open` and `Close` prices, the strategy determines whether to take a long or short position at market open and closes the trade by day’s end.

### Trading Rules

1. **Positive Overnight Return → Long Position**
   - If the current day’s `Open` is higher than the previous day’s `Close`, go long by investing $100 at the `Open` price and selling at the `Close`.
   - Profit/Loss per share: `Close – Open`.

2. **Negative Overnight Return → Short Position**
   - If the current day’s `Open` is lower than the previous day’s `Close`, short $100 worth of shares at `Open` and cover at `Close`.
   - Profit/Loss per share: `Open – Close`.

3. **Assumptions**
   - The strategy trades every day unless `Open` equals the previous day’s `Close`.  
   - Each trade uses a fixed $100 investment.  
   - Transaction costs are ignored for simplicity.


---

## 2. Weekly Machine Learning Strategy (BA)

This section builds on the same concept but scales it to a weekly decision framework using machine learning.  
Weekly Boeing data is used to compute:

- **Mean Return (μ):** average weekly price change  
- **Volatility (σ):** standard deviation of weekly returns  

Each week is labeled as:
- **Green** → buy signal (stay invested)  
- **Red** → sell signal (move to cash)  

### Train/Test Split
- **Training Period:** 2023–2024  
- **Testing Period:** 2020–2022  

### Linear Classification Baseline
A simple visual rule from 2023 showed that a vertical cutoff at **μ = −100** cleanly separates green and red weeks.  
Applied to 2024, this rule achieved **100% accuracy** and produced a **$162.39** profit from a $100 initial investment.

### Machine Learning Models
- k-Nearest Neighbors (kNN)  
- Logistic Regression  
- Naïve Bayes (Student-t and Exponential)  
- LDA / QDA  
- Decision Tree  
- Random Forest  
- Support Vector Machines (Linear, Gaussian/RBF, Polynomial degree 2)

### Comparison of Classification Models

| Model           |   TPR | TNR | Accuracy | Trading Strategy (Year 1–2023) | Trading Strategy (Year 2–2024) |
|-----------------|-------|-----|----------|---------------------------------|---------------------------------|
| kNN             | 0.812 | 1   | 0.971    | 171.47                          | 252.42                          |
| Logreg          | 0.688 | 1   | 0.952    | 152.69                          | 224.78                          |
| NB              | 0.188 | 1   | 0.876    | 135.09                          | 118.84                          |
| NB (student-t)  | 0     | 1   | 0.848    | 135.09                          | 91.50                           |
| NB (exp)        | 1     | 0   | 0.152    | 100.00                          | 100.00                          |
| LDA             | 0.188 | 1   | 0.876    | 135.09                          | 133.51                          |
| QDA             | 0.188 | 1   | 0.876    | 135.09                          | 133.51                          |
| Decision Tree   | 1     | 1   | 1.000    | 181.28                          | 294.39                          |
| Random Forest   | 1     | 1   | 1.000    | 181.28                          | 294.39                          |
| Linear SVM      | 0.750 | 1   | 0.962    | 159.58                          | 234.92                          |
| Gaussian SVM    | 0.812 | 1   | 0.971    | 159.58                          | 245.92                          |
| Poly SVM (d=2)  | 0     | 1   | 0.848    | 135.09                          | 91.50                           |

**Summary of Findings:**  
- **Best Accuracy (97.1%)**: kNN and Gaussian SVM  
- **Best Trading Performance**: Decision Tree and Random Forest (up to **$294.39**)  
- Naïve Bayes (exponential) failed to enter trades (TPR = 0), giving the lowest accuracy (15.2%).  
- Linear SVM, Logistic Regression, and Gaussian SVM showed good balance between accuracy and profit.  
- Probabilistic models struggled due to correlation between features.  
- Polynomial SVM (d=2) overfit the data.  

**Key Learning:**  
Logistic Regression requires minimal tuning, whereas kNN, Random Forest, and SVM depend heavily on scaling and hyperparameters. Naïve Bayes performs poorly on correlated features, and increasing polynomial degree in SVM leads to overfitting.


---

## 3. Clustering Analysis

Beyond supervised classification, unsupervised learning techniques were applied to explore structure in weekly return–volatility patterns.

### K-Means Clustering on Boeing (BA)
- Weekly (μ, σ) features were clustered using k-means.  
- The elbow method suggested **k = 4** as the optimal number of clusters.  
- Some clusters contained mostly “green” weeks, others mostly “red,” showing clear behavioral regimes.  
- Cluster purity analysis was used to measure how consistently each cluster aligned with trading labels.

Results showed that return–volatility features naturally form meaningful regimes, further supporting the ML findings. This provided an unsupervised confirmation that BA’s weekly behavior is predictable and forms stable patterns over time.

### Clustering Across Dow Jones Stocks
The same k-means workflow was applied to five Dow Jones components: **AMZN, JNJ, MCD, NKE, NVDA**.  
Cluster trajectories were tracked month-to-month to compare stability and similarity.

#### Hamming Distance Analysis
- **Most different behavior:** NVDA vs. JNJ (distance = 50)  
- **Most similar behavior:** MCD vs. NKE (distance = 28)  
- **Most stable:** MCD and NKE (long cluster streaks)  
- **Least stable:** NVDA (frequent cluster switching)  

This analysis highlights cross-stock structural relationships not visible in raw price charts.


---

## Tools and Libraries
- **Python**, **Pandas**, **NumPy**
- **scikit-learn** – machine learning and clustering models  
  - Classification: KNN, Logistic Regression, SVM (Linear, RBF, Polynomial), Naïve Bayes, LDA/QDA, Decision Tree, Random Forest, AdaBoost  
  - Clustering: KMeans  
  - Preprocessing: StandardScaler  
- **SciPy** – Hamming distance calculations  
- **Matplotlib**, **Seaborn** – visualization  
- **Jupyter Notebook**, **Quarto** – analysis and reporting
