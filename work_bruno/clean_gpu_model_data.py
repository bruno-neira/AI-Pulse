import pandas as pd
import re

# Load datasets
gpu_clusters = pd.read_csv('../Data/gpu_clusters_dataset/gpu_clusters.csv')
notable_models = pd.read_csv('../Data/ai_models_dataset/notable_ai_models.csv')

# Select and rename relevant columns for GPU clusters
gpu_data = gpu_clusters[['Name', 'Owner', 'First Operational Date']].copy()
gpu_data.columns = ['name', 'owner', 'date']
gpu_data['type'] = 'cluster'
gpu_data['training_compute'] = pd.NA
gpu_data['total_compute_available'] = pd.NA

# Select and rename relevant columns for AI models
model_data = notable_models[['Model', 'Organization', 'Publication date', 'Training compute (FLOP)']].copy()
model_data.columns = ['name', 'owner', 'date', 'training_compute']
model_data['type'] = 'model'
model_data['total_compute_available'] = pd.NA

# Combine datasets
combined_data = pd.concat([model_data, gpu_data], ignore_index=True)

# Clean owner names in GPU clusters to match model organizations
def clean_owner_name(owner):
    if pd.isna(owner):
        return owner

    # Extract organization name from parentheses if present
    # e.g., "Stargate (OpenAI)" -> "OpenAI"
    paren_match = re.search(r'\(([^)]+)\)', owner)
    if paren_match:
        return paren_match.group(1)

    # Remove common suffixes and clean up
    owner = owner.strip()

    # Handle comma-separated multiple owners - take first one for simplicity
    # or you could expand this to create multiple rows
    if ',' in owner:
        owner = owner.split(',')[0].strip()

    # Common mappings
    mappings = {
        'Microsoft': 'Microsoft',
        'Meta': 'Meta',
        'Google': 'Google DeepMind',  # Match to Google DeepMind if that's common in models
        'Amazon': 'Amazon',
        'OpenAI': 'OpenAI',
        'Anthropic': 'Anthropic',
        'Alibaba': 'Alibaba',
        'Tesla': 'Tesla',
        'xAI': 'xAI',
        'Oracle': 'Oracle',
        'Tencent': 'Tencent'
    }

    # Check if owner contains any of the key organization names
    for key, value in mappings.items():
        if key.lower() in owner.lower():
            return value

    return owner

# Apply cleaning to GPU cluster owners
mask_cluster = combined_data['type'] == 'cluster'
combined_data.loc[mask_cluster, 'owner'] = combined_data.loc[mask_cluster, 'owner'].apply(clean_owner_name)

# Convert date columns to datetime
combined_data['date'] = pd.to_datetime(combined_data['date'], errors='coerce')

# Convert training_compute to numeric
combined_data['training_compute'] = pd.to_numeric(combined_data['training_compute'], errors='coerce')

# Save the cleaned combined dataset
combined_data.to_csv('../Data/combined_gpu_models.csv', index=False)

print("Data cleaning complete!")
print(f"Total rows: {len(combined_data)}")
print(f"Models: {len(combined_data[combined_data['type'] == 'model'])}")
print(f"Clusters: {len(combined_data[combined_data['type'] == 'cluster'])}")
print(f"\nUnique owners in models: {combined_data[combined_data['type'] == 'model']['owner'].nunique()}")
print(f"Unique owners in clusters: {combined_data[combined_data['type'] == 'cluster']['owner'].nunique()}")
print(f"\nSample of cluster owners:\n{combined_data[combined_data['type'] == 'cluster']['owner'].value_counts().head(10)}")
