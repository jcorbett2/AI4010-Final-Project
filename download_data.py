import kagglehub

# Download latest version
leaf_path = kagglehub.competition_download('leaf-classification')

print("Path to competition files:", leaf_path)

# Download latest version
documents_path = kagglehub.competition_download('denoising-dirty-documents')

print("Path to competition files:", documents_path)