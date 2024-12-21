import kagglehub
# Download latest version
path = kagglehub.dataset_download("ryandpark/fruit-quality-classification")
print("Path to dataset files:", path)