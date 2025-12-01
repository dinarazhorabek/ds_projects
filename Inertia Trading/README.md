# Inertia Trading – Boeing (BA) & S&P 500 (SPY)
### Machine Learning, Rule-Based Strategies, and Clustering Analysis

## Overview

This repository explores short-term trading behavior in Boeing (BA) using a combination of rule-based trading, supervised machine learning, and unsupervised clustering.  
SPY (S&P 500 ETF) is included as a baseline comparison in the initial analysis.

The project consists of three main components:
1. Day Trading – Inertia Strategy (BA & SPY)  
   A rule-based approach using daily open–close behavior to test whether overnight price movements persist intraday.  
   Positions are opened at the market open and closed at the same day’s close, using a fixed $100 per trade.

2. Weekly Trading – Machine Learning Labeling Strategy (BA)  
   A classification-based approach using weekly features such as mean return and volatility.  
   Multiple ML models—including Logistic Regression, kNN, SVMs, Decision Trees, Random Forest, Naïve Bayes, and LDA/QDA—are trained (2023–2024) and tested out-of-sample (2020–2022) to label weeks as  
   **green (buy)** or **red (cash)** and evaluate trading performance relative to buy-and-hold.

3. Clustering Analysis (BA + Dow Jones Stocks)  
   Weekly (µ, σ) patterns are also explored using k-means clustering to identify natural behavioral regimes.  
   Additional clustering of Dow Jones stocks (AMZN, JNJ, MCD, NKE, NVDA) is used to compare cross-stock behavior using cluster trajectories and Hamming distances to measure similarity and stability.


## 1. Day Trading – Inertia Strategy

### Concept
The inertia approach assumes that overnight price changes often persist during the following trading day.  
Using daily `Open` and `Close` prices, the strategy decides whether to take a long or short position at market open and closes the trade by day’s end.

### Trading Rules

1. **Positive Overnight Return → Long Position**
   - If the current day’s `Open` is higher than the previous day’s `Close`,  
     go long by investing $100 at the `Open` price and selling at the `Close`.
   - Profit/Loss per share: `Close – Open`

2. **Negative Overnight Return → Short Position**
   - If the current day’s `Open` is lower than the previous day’s `Close`,  
     short $100 worth of shares at `Open` and cover at `Close`.
   - Profit/Loss per share: `Open – Close`

3. **Assumptions**
   - The strategy trades every day unless `Open` equals the previous day’s `Close`.
   - Each trade uses a fixed $100 investment.
   - Transaction costs are ignored for simplicity.


## 2. Weekly Machine Learning Strategy (BA)

This section builds on the same concept but scales it to a weekly decision framework using machine learning.  
It uses historical Boeing data to compute:

- Mean Return (μ): average weekly price change  
- Volatility (σ): standard deviation of weekly returns  

Each week is labeled as:
- Green → buy signal (stay invested)  
- Red → sell signal (move to cash)  

Train/Test Split:
- Training Period: 2023-2024
- Testing Period: 2020-2022

### Linear Classification Baseline

A simple visual rule from 2023 showed that a vertical cutoff at μ = −100 cleanly separates green and red weeks.  
Applied to 2024, this rule achieved 100 % accuracy and generated a $162.39 profit from a $100 initial investment.

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

Based on the results, the kNN and Gaussian SVM models achieved the highest accuracy
(97.1%), effectively identifying both “green” (buy) and “red” (sell) weeks, with true positive
rates around 81%. From a trading perspective, the Decision Tree and Random Forest strategies
produced the highest cumulative earnings across both years ($294.39), showing consistent
performance and stable predictions.

The kNN model also performed strongly, reaching $252.42 in 2024, confirming its robustness in
pattern recognition. In contrast, the Naïve Bayes (exponential) model failed to enter trades,
maintaining the initial $100 cash balance with the lowest accuracy (15.2%). This suggests that
the exponential density function was unsuitable for this dataset.

Models like Logistic Regression, Linear SVM, and Gaussian SVM achieved a good balance
between accuracy and trading performance, while LDA, QDA, and Naïve Bayes (Student-t)
showed moderate accuracy (~87%) but weaker profit outcomes like a buy-and-hold strategy.
Overall, these findings indicate that the data can be effectively separated and clustered, likely
due to strong correlations between features (mean return and volatility). Tree-based and kernel-
based models outperformed probabilistic ones, making them more reliable for this stock’s trend
prediction.

While implementing all these models, I gained a deeper understanding of how each algorithm
works and what differentiates them. For example, logistic regression is relatively simple and
doesn’t require tuning hyperparameters, unlike models such as kNN or Random Forest. Both
kNN and SVM models require feature scaling since they rely on distance-based calculations. I
also learned that Naïve Bayes tends to perform poorly on datasets with highly correlated
features. Additionally, increasing the polynomial degree in SVM models can lead to overfitting,
where the model captures noise instead of meaningful patterns.

## 3. Clustering Analysis
Beyond supervised classification, unsupervised learning techniques were applied to explore structure in weekly return–volatility patterns.

### K-Means Clustering on Boeing (BA)
- Weekly features (mean return µ and volatility σ) for BA were clustered using k-means to identify natural behavioral regimes.
- The optimal number of clusters was chosen using the inertia (elbow) method, which suggested k = 4 as the best balance between separation and complexity.
- Each cluster showed distinct characteristics—some dominated by “green” (buy) weeks, others by “red” (cash) weeks.
- Cluster purity analysis was used to measure how consistently each cluster aligned with trading labels.

Results showed that return–volatility features naturally form meaningful regimes, further supporting the ML findings. This provided an unsupervised confirmation that BA’s weekly behavior is predictable and forms stable patterns over time.

### Clustering Across Dow Jones Stocks
To compare the stability and similarity of stock behavior, the same k-means clustering framework was applied to five Dow Jones components: AMZN, JNJ, MCD, NKE, NVDA.

Each stock’s weekly (µ, σ) values were clustered, and we tracked how the stock moved across clusters month-to-month.

**Hamming Distance Analysis**

A Hamming distance metric was used to compare cluster trajectories between stocks:
- Highest distance:
   NVDA vs. JNJ (distance = 50) → most different behavioral patterns
- Lowest distance:
   MCD vs. NKE (distance = 28) → most similar trajectories
- Stability:
   MCD and NKE had the longest cluster streaks, indicating steady behavior
- Variability:
   NVDA showed the highest month-to-month switching, making it the least stable

This analysis demonstrated how stocks with very different market profiles can be quantified using unsupervised learning, highlighting relationships not easily visible through price charts alone.

## Tools and Libraries
- Python, Pandas, NumPy
- scikit-learn – machine learning and clustering models  
  - Classification: KNN, Logistic Regression, SVM (Linear, RBF, Polynomial), Naïve Bayes, LDA/QDA, Decision Tree, Random Forest, AdaBoost  
  - Clustering: KMeans  
  - Preprocessing: StandardScaler 
- SciPy – Hamming distance calculations for trajectory comparison  
- Matplotlib, Seaborn
- Jupyter Notebook / Quarto