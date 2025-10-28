import os

# List of model files to delete
files_to_delete = [
    'decision_tree_model.pkl',
    'LinearSVC.pkl',
    'Logistic_regression_model.pkl',
    'naive_bayes_model.pkl',
    'standard_scalar.pkl',
    'svc_model.pkl',
]

production_dir = os.path.join(os.path.dirname(__file__), 'production')
for filename in files_to_delete:
    file_path = os.path.join(production_dir, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted: {filename}")
    else:
        print(f"Not found: {filename}")
print("All old model files deleted.")
