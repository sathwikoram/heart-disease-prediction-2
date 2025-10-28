import os
import pickle
import joblib

config = {
    'heart': {
        'SVC': 'production/svc_model.pkl',
        'LogisticRegression': 'production/Logistic_regression_model.pkl',
        'NaiveBayes': 'production/naive_bayes_model.pkl',
        'DecisionTree': 'production/decision_tree_model.pkl',
        'scalar_file': 'production/standard_scalar.pkl',
    }}

dir = os.path.dirname(__file__)

def GetJobLibFile(filepath):
    if os.path.isfile(os.path.join(dir, filepath)):
        return joblib.load(os.path.join(dir, filepath))
    else:
        print("file does not exit")

def GetPickleFile(filepath):
    full_path = os.path.join(dir, filepath)
    if os.path.isfile(full_path):
        with open(full_path, "rb") as f:
            return pickle.load(f)
    else:
        print(f"file does not exist: {full_path}")
    return None

def GetStandardScalarForHeart():
    scalar = GetPickleFile(config['heart']['scalar_file'])
    if scalar is None:
        raise FileNotFoundError(f"StandardScaler file not found or could not be loaded: {os.path.join(dir, config['heart']['scalar_file'])}")
    return scalar

def GetAllClassifiersForHeart():
    return (GetSVCClassifierForHeart(),GetLogisticRegressionClassifierForHeart(),GetNaiveBayesClassifierForHeart(),GetDecisionTreeClassifierForHeart())

def GetSVCClassifierForHeart():
    return GetJobLibFile(config['heart']['SVC'])

def GetLogisticRegressionClassifierForHeart():
    return GetJobLibFile(config['heart']['LogisticRegression'])

def GetNaiveBayesClassifierForHeart():
    return GetJobLibFile(config['heart']['NaiveBayes'])

def GetDecisionTreeClassifierForHeart():
    return GetJobLibFile(config['heart']['DecisionTree'])
