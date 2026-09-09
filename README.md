# JPMorgan Quantitative Research Projects

This repository contains projects completed as part of JPMorgan Chase & Co.’s Quantitative Research Job Simulation via Forage. These projects focus on key aspects of financial analysis, quantitative modeling, and risk assessment.

# 📌 Projects Overview

1️⃣ Investigate and Analyze Price Data
	
•	Explored market price data to identify trends and insights.

•	Implemented statistical techniques to analyze financial time series data.

2️⃣ Price a Commodity Storage Contract
	
•	Modeled the pricing of storage contracts for commodities.

•   Applied financial engineering techniques to assess market behavior.

3️⃣ Credit Risk Analysis

•	Evaluated credit risk using financial data.

•	Developed models to assess and predict default probabilities.

4️⃣ Bucket FICO Scores
	
•	Categorized FICO scores into risk buckets.
	
•	Analyzed creditworthiness and its impact on financial products.

# 🧭 Module → Task Map

| Module | Forage task | Reads | Produces |
|---|---|---|---|
| `src/naturalgas.py` | 1 - Investigate and analyze price data | `Nat_Gas.csv` (`Date`, `Price`); bundled equivalent is `datasets/naturalgas.csv`, which uses `Dates`/`Prices` | Price & forecast plots, plus a 12-month Holt-Winters forecast and an `estimate_price(date)` lookup |
| `src/naturalgaspricing.py` | 2 - Price a commodity storage contract | `Nat_Gas.csv` (`Date`, `Price`); bundled equivalent is `datasets/naturalgas.csv` | Net cash flow of a gas storage contract from injection/withdrawal schedules, rates, max volume and storage cost |
| `src/loanmodel.py` | 3 - Credit risk analysis | `loan_data.csv` (`income`, `loan_amount`, `credit_score`, `default`); bundled equivalent is `datasets/customerloan.csv`, which uses `loan_amt_outstanding`/`fico_score` | Logistic-regression probability of default and expected loss (10% recovery rate) |
| `src/ficoBucketQuant.py` | 4 - Bucket FICO scores | Nothing - it generates random scores and default counts in-file | FICO buckets by equal-size/MSE split; the log-likelihood optimiser currently raises `KeyError` (see the module docstring) |

> Note: the scripts have hard-coded input filenames (`Nat_Gas.csv`, `loan_data.csv`) that do not match the files in `datasets/`, and the bundled CSVs use different column names. Update the path and column names before running a script against the bundled data.

# 🛠️ Tech Stack
	
•	Python (Pandas, NumPy, Scikit-learn)

•	Financial Modeling Techniques

# 📂 Repository Structure
```
JPMorgan-Quant-Projects/
│── datasets/
│   ├── customerloan.csv
│   ├── naturalgas.csv
│── src/
│   ├── ficoBucketQuant.py
│   ├── loanmodel.py
│   ├── naturalgas.py
│   ├── naturalgaspricing.py
│── LICENSE
│── README.md
```
# 🚀 Getting Started
1. Clone the repository:
```  
git clone https://github.com/myselfRaifMondal/JPMorgan-Quant-Projects.git
```
2.	Navigate into a project directory and explore scripts.

# 🎯 About

This repository showcases my work in Quantitative Research, covering data analysis, financial modeling, and risk assessment. These projects demonstrate key skills required for roles in Quantitative Finance, Risk Management, and Data Science.