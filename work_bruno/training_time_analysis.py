import pandas as pd
import numpy as np

# Load datasets
print("Loading datasets...")
frontier_models = pd.read_csv('../Data/ai_models_dataset/frontier_ai_models.csv')
gpu_clusters = pd.read_csv('../Data/gpu_clusters_dataset/gpu_clusters.csv')

print(f"Initial frontier models: {len(frontier_models)}")
print(f"Initial GPU clusters: {len(gpu_clusters)}")

# Filter frontier models to only those with known training compute
frontier_models_filtered = frontier_models[
    frontier_models['Training compute (FLOP)'].notna()
].copy()

print(f"\nFrontier models with known training compute: {len(frontier_models_filtered)}")

# Convert training compute to numeric and calculate log
frontier_models_filtered['Training compute (FLOP)'] = pd.to_numeric(
    frontier_models_filtered['Training compute (FLOP)'],
    errors='coerce'
)
frontier_models_filtered['Training compute (log)'] = np.log10(
    frontier_models_filtered['Training compute (FLOP)']
)

# Convert publication date to datetime
frontier_models_filtered['Publication date'] = pd.to_datetime(
    frontier_models_filtered['Publication date'],
    errors='coerce'
)

# Display sample of frontier models
print("\nSample of frontier models:")
print(frontier_models_filtered[['Model', 'Organization', 'Publication date', 'Training compute (FLOP)', 'Training compute (log)']].head())

# Filter GPU clusters to only rank=1 clusters with known 16-bit OP/s
gpu_clusters_filtered = gpu_clusters[
    (gpu_clusters['Rank when first operational'] == 1) &
    (gpu_clusters['16-bit OP/s (log)'].notna())
].copy()

print(f"\nRank-1 GPU clusters with known 16-bit OP/s: {len(gpu_clusters_filtered)}")

# Convert first operational date to datetime
gpu_clusters_filtered['First Operational Date'] = pd.to_datetime(
    gpu_clusters_filtered['First Operational Date'],
    errors='coerce'
)

# Convert 16-bit OP/s to numeric
gpu_clusters_filtered['16-bit OP/s (log)'] = pd.to_numeric(
    gpu_clusters_filtered['16-bit OP/s (log)'],
    errors='coerce'
)

# Display sample of GPU clusters
print("\nSample of rank-1 GPU clusters:")
print(gpu_clusters_filtered[['Name', 'Owner', 'First Operational Date', '16-bit OP/s (log)']].head())

# Get top 10 models by training compute
top_10_models = frontier_models_filtered.nlargest(10, 'Training compute (FLOP)')
print("\n=== Top 10 Models by Training Compute ===")
print(top_10_models[['Model', 'Organization', 'Publication date', 'Training compute (FLOP)']].to_string())

# Calculate average training compute of top 10 models (in log scale)
avg_training_compute_log = top_10_models['Training compute (log)'].mean()
avg_training_compute = top_10_models['Training compute (FLOP)'].mean()

print(f"\n=== Average Training Compute of Top 10 Models ===")
print(f"Average (log): {avg_training_compute_log:.2f}")
print(f"Average (FLOP): {avg_training_compute:.2e}")

# Calculate training time for each rank-1 cluster
# Training time (seconds) = Training compute (FLOP) / Compute per second (FLOP/s)
# In log scale: log(training_time) = log(training_compute) - log(compute_per_second)

gpu_clusters_filtered['Training time (log seconds)'] = avg_training_compute_log - gpu_clusters_filtered['16-bit OP/s (log)']

# Convert from log seconds to actual seconds, then to hours
gpu_clusters_filtered['Training time (seconds)'] = 10 ** gpu_clusters_filtered['Training time (log seconds)']
gpu_clusters_filtered['Training time (minutes)'] = gpu_clusters_filtered['Training time (seconds)'] / 60
gpu_clusters_filtered['Training time (hours)'] = gpu_clusters_filtered['Training time (minutes)'] / 60
gpu_clusters_filtered['Training time (days)'] = gpu_clusters_filtered['Training time (hours)'] / 24

print("\n=== Training Time for Each Rank-1 Cluster ===")
print("(Time to train average of top 10 frontier models)")
result_cols = ['Name', 'Owner', 'First Operational Date', '16-bit OP/s (log)',
               'Training time (hours)', 'Training time (days)']
print(gpu_clusters_filtered[result_cols].sort_values('First Operational Date').to_string())

# Summary statistics
print("\n=== Summary Statistics ===")
print(f"Median training time: {gpu_clusters_filtered['Training time (hours)'].median():.2f} hours")
print(f"Mean training time: {gpu_clusters_filtered['Training time (hours)'].mean():.2f} hours")
print(f"Min training time: {gpu_clusters_filtered['Training time (hours)'].min():.2f} hours")
print(f"Max training time: {gpu_clusters_filtered['Training time (hours)'].max():.2f} hours")

print("\nDataframes created successfully!")
print("- frontier_models_filtered: Frontier models with known training compute")
print("- gpu_clusters_filtered: Rank-1 clusters with calculated training times")
