"""Task 3 - Credit risk: probability of default and expected loss (JPMorgan Forage).

Trains a scikit-learn ``LogisticRegression`` (on ``StandardScaler``-scaled
features, 80/20 train/test split, ``random_state=42``) to predict whether a
borrower defaults, then exposes:

* ``predict_default_probability(loan_details)`` - probability of default for a
  single ``[income, loan_amount, credit_score]`` record.
* ``expected_loss(loan_details, loan_amount)`` - that probability multiplied by
  the loss given default, assuming a fixed 10% recovery rate.

Input: ``pd.read_csv('loan_data.csv')`` with the columns ``income``,
``loan_amount``, ``credit_score`` and ``default``. NOTE: that file is not in
this repository and the path is hard-coded, so the script fails with
``FileNotFoundError`` as written. The equivalent bundled data is
``datasets/customerloan.csv``, whose columns are named differently
(``income``, ``loan_amt_outstanding``, ``fico_score``, ``default``), so the
path and the column list both have to be edited before it will run.

Output: prints the probability of default and the expected loss for the
hard-coded sample loan ``[50000, 20000, 700]``.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset (replace 'loan_data.csv' with actual file path)
df = pd.read_csv('loan_data.csv')

# Assume the dataset has columns: ['income', 'loan_amount', 'credit_score', 'default']
X = df[['income', 'loan_amount', 'credit_score']]
y = df['default']

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict probability of default
def predict_default_probability(loan_details):
    loan_details = np.array(loan_details).reshape(1, -1)
    loan_details = scaler.transform(loan_details)
    return model.predict_proba(loan_details)[0][1]

# Calculate expected loss
def expected_loss(loan_details, loan_amount):
    pd = predict_default_probability(loan_details)
    recovery_rate = 0.1  # 10% recovery rate
    loss_given_default = loan_amount * (1 - recovery_rate)
    return pd * loss_given_default

# Example usage
sample_loan = [50000, 20000, 700]  # income, loan amount, credit score
loan_amount = 20000
print("Probability of Default:", predict_default_probability(sample_loan))
print("Expected Loss:", expected_loss(sample_loan, loan_amount))

