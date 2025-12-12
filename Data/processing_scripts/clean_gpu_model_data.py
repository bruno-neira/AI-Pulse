"""
Data Cleaning: GPU Clusters and AI Models
This script combines GPU cluster data with notable AI models data into a single dataset for visualization.
"""

import pandas as pd
import re
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.dirname(script_dir)  # Parent directory (Data/)

print("Starting GPU and Model data cleaning process...")

# ============================================================================
# Load Datasets
# ============================================================================
print("\n=== Loading Datasets ===")

gpu_clusters = pd.read_csv(os.path.join(data_dir, 'gpu_clusters_dataset', 'gpu_clusters.csv'))
notable_models = pd.read_csv(os.path.join(data_dir, 'ai_models_dataset', 'notable_ai_models.csv'))

print(f"GPU Clusters shape: {gpu_clusters.shape}")
print(f"Notable Models shape: {notable_models.shape}")

# ============================================================================
# Prepare GPU Cluster Data
# ============================================================================
print("\n=== Preparing GPU Cluster Data ===")

# Select and rename relevant columns for GPU clusters
# Only include existing clusters
gpu_data = gpu_clusters[gpu_clusters['Status'] == 'Existing'][['Name', 'Owner', 'First Operational Date', 'H100 equivalents']].copy()
gpu_data.columns = ['name', 'owner', 'date', 'total_compute_available']
gpu_data['type'] = 'cluster'
gpu_data['training_compute'] = pd.NA

print(f"Total existing GPU clusters: {len(gpu_data)}")

# ============================================================================
# Prepare AI Model Data
# ============================================================================
print("\n=== Preparing AI Model Data ===")

# Select and rename relevant columns for AI models
model_data = notable_models[['Model', 'Organization', 'Publication date', 'Training compute (FLOP)']].copy()
model_data.columns = ['name', 'owner', 'date', 'training_compute']
model_data['type'] = 'model'
model_data['total_compute_available'] = pd.NA

# ============================================================================
# Combine Datasets
# ============================================================================
print("\n=== Combining Datasets ===")

combined_data = pd.concat([model_data, gpu_data], ignore_index=True)
print(f"Combined dataset shape: {combined_data.shape}")

# ============================================================================
# Clean Owner Names
# ============================================================================
print("\n=== Cleaning Owner Names ===")

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

    # Handle comma-separated multiple owners
    # Prioritize AI-first companies if present
    if ',' in owner:
        owners_list = [o.strip() for o in owner.split(',')]

        # AI-first companies to prioritize (in order of priority)
        ai_first_companies = ['OpenAI', 'Anthropic', 'xAI', 'Meta', 'Google', 'DeepMind',
                              'Alibaba', 'Mistral', 'Cohere', 'Inflection']

        # Check if any AI-first company is in the list
        for ai_company in ai_first_companies:
            for o in owners_list:
                if ai_company.lower() in o.lower():
                    owner = o
                    break
            else:
                continue
            break
        else:
            # No AI-first company found, take first owner
            owner = owners_list[0]

    # Common mappings to standardize names
    mappings = {
        'Microsoft': 'Microsoft',
        'Meta AI': 'Meta',  # Map Meta AI to Meta first
        'Meta': 'Meta',
        'Google DeepMind': 'Google DeepMind',  # Check full name first
        'DeepMind': 'Google DeepMind',  # Map DeepMind to Google DeepMind
        'Google': 'Google DeepMind',
        'Amazon': 'Amazon',
        'OpenAI': 'OpenAI',
        'Anthropic': 'Anthropic',
        'Ant Group': 'Alibaba',  # Ant Group is a subsidiary of Alibaba
        'Alibaba': 'Alibaba',
        'Tesla': 'Tesla',
        'xAI': 'xAI',
        'Oracle': 'Oracle',
        'Tencent': 'Tencent',
        'Mistral': 'Mistral',
        'Cohere': 'Cohere',
        'Inflection': 'Inflection AI'
    }

    # Check if owner contains any of the key organization names
    # Check for exact matches first, then partial matches
    for key, value in mappings.items():
        if owner.lower() == key.lower():
            return value

    for key, value in mappings.items():
        if key.lower() in owner.lower():
            return value

    return owner

# Apply cleaning to GPU cluster owners
mask_cluster = combined_data['type'] == 'cluster'
combined_data.loc[mask_cluster, 'owner'] = combined_data.loc[mask_cluster, 'owner'].apply(clean_owner_name)

# ============================================================================
# Data Type Conversions and Filtering
# ============================================================================
print("\n=== Converting Data Types and Filtering ===")

# Convert date columns to datetime
combined_data['date'] = pd.to_datetime(combined_data['date'], errors='coerce')

# Convert training_compute to numeric
combined_data['training_compute'] = pd.to_numeric(combined_data['training_compute'], errors='coerce')

# Filter models to only include those with known training compute values
mask_model = combined_data['type'] == 'model'
mask_cluster = combined_data['type'] == 'cluster'

# Keep only models with non-null training_compute
models_with_compute = combined_data[mask_model & combined_data['training_compute'].notna()].copy()
all_clusters = combined_data[mask_cluster].copy()

print(f"Models with known training compute: {len(models_with_compute)}")
print(f"Total clusters: {len(all_clusters)}")

# Recombine
combined_data = pd.concat([models_with_compute, all_clusters], ignore_index=True)

# Filter to only include models/clusters with mutual ownership
# Get unique owners from each type
model_owners = set(combined_data[combined_data['type'] == 'model']['owner'].dropna().unique())
cluster_owners = set(combined_data[combined_data['type'] == 'cluster']['owner'].dropna().unique())

# Find intersection - owners that have both models and clusters
mutual_owners = model_owners & cluster_owners

print(f"\nOwners with both models (with compute) and existing clusters: {len(mutual_owners)}")
print(f"Mutual owners: {sorted(mutual_owners)}")

# Filter combined data to only include these mutual owners
combined_data = combined_data[combined_data['owner'].isin(mutual_owners)].copy()

print(f"\nAfter filtering:")
print(f"Total rows: {len(combined_data)}")
print(f"Models: {len(combined_data[combined_data['type'] == 'model'])}")
print(f"Clusters: {len(combined_data[combined_data['type'] == 'cluster'])}")

# ============================================================================
# Final Dataset Summary
# ============================================================================
print("\n=== Final Dataset Summary ===")
print(f"Total rows: {len(combined_data)}")
print(f"Models: {len(combined_data[combined_data['type'] == 'model'])}")
print(f"Clusters: {len(combined_data[combined_data['type'] == 'cluster'])}")
print(f"\nUnique owners: {combined_data['owner'].nunique()}")
print(f"\nOwner distribution:")
print(combined_data.groupby('owner')['type'].value_counts().unstack(fill_value=0))

# ============================================================================
# Save Combined Dataset
# ============================================================================
output_path = os.path.join(data_dir, 'combined_gpu_models.csv')
combined_data.to_csv(output_path, index=False)
print(f"\n=== Data saved to {output_path} ===")
