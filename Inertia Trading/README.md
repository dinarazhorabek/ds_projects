# Inertia Trading – Boeing (BA) & S&P 500 (SPY)
### Machine Learning and Rule-Based Short-Term Trading Strategies

## 🧭 Overview

This repository explores short-term trading strategies using Boeing (BA) and, for comparison, the S&P 500 ETF (SPY).  
Two distinct but complementary approaches are developed:

1. Day Trading – Inertia Strategy (BA & SPY)  
   Focuses on intraday momentum persistence using daily open–close prices.  
   It tests whether overnight price movements tend to continue during the following trading day.  
   SPY data is included only for baseline comparison at the beginning of the analysis.  

2. Weekly Trading – Machine Learning Labeling Strategy (BA)  
   Extends the analysis to a weekly horizon using features such as mean return and volatility.  
   Employs k-Nearest Neighbors (k-NN) and Logistic Regression models to classify weeks as green (buy) or red (cash)  
   and simulate portfolio performance relative to a buy-and-hold baseline.

---

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

---

## 2. Weekly Machine Learning Strategy (BA)

This section builds on the same concept but scales it to a weekly decision framework using machine learning.  
It uses historical Boeing data to compute:

- Mean Return (μ): average weekly price change  
- Volatility (σ): standard deviation of weekly returns  

Each week is labeled as:
- Green → buy signal (stay invested)  
- Red → sell signal (move to cash)  

Models:
- k-NN (k=3): nonlinear classifier achieving 97% accuracy  
- Logistic Regression: interpretable linear model with 94% accuracy  

Both models significantly outperform buy-and-hold when simulated over 2023–2024.

---

## Tools and Libraries
- Python, Pandas, NumPy  
- scikit-learn (KNeighborsClassifier, LogisticRegression)  
- Matplotlib, Seaborn  
- Jupyter Notebook / Quarto

---

## Author
Dinara Zhorabek  
Graduate Student in Applied Business Analytics, Boston University  
📧 [LinkedIn Profile](https://www.linkedin.com/in/dinarazhorabek/)