import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np

# Dummy data: 10 samples, 13 features (adjust shape if needed)
dummy_data = np.random.rand(10, 13)
scaler = StandardScaler()
scaler.fit(dummy_data)

with open('c:\\Projects\\Heart-disease-prediction\\predict_risk\\production\\standard_scalar.pkl', 'wb') as f:
    pickle.dump(scaler, f)