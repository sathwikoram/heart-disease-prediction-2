import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Replace this with your real training data!
X_train = np.random.rand(100, 13)  # 100 samples, 13 features
y_train = np.random.randint(0, 2, 100)  # Binary target

# Fit scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Fit SVC
svc = SVC(probability=True)
svc.fit(X_scaled, y_train)

# Fit Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_scaled, y_train)

# Fit Naive Bayes
nb = GaussianNB()
nb.fit(X_scaled, y_train)

# Fit Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_scaled, y_train)

# Ensure production directory exists
production_dir = os.path.join(os.path.dirname(__file__), 'production')
os.makedirs(production_dir, exist_ok=True)

# Save scaler
with open(os.path.join(production_dir, 'standard_scalar.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
print("standard_scalar.pkl has been saved in the production folder.")

# Save SVC model
with open(os.path.join(production_dir, 'svc_model.pkl'), 'wb') as f:
    pickle.dump(svc, f)

# Save Logistic Regression model
joblib.dump(logreg, os.path.join(production_dir, 'Logistic_regression_model.pkl'))

# Save Naive Bayes model
joblib.dump(nb, os.path.join(production_dir, 'naive_bayes_model.pkl'))

# Save Decision Tree model
joblib.dump(dt, os.path.join(production_dir, 'decision_tree_model.pkl'))

print("All models and scaler have been saved in the production folder.")