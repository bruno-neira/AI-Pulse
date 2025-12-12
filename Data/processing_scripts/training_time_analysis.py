"""
Training Time Analysis
Calculate how long it would take to train the average of the top 10 frontier models
on each rank-1 GPU cluster.
"""

import pandas as pd
import numpy as np
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.dirname(script_dir)  # Parent directory (Data/)

print("=" * 60)
print("Training Time Analysis")
print("=" * 60)

# ============================================================================
# Load Datasets
# ============================================================================
print("\nLoading datasets...")
frontier_models = pd.read_csv(os.path.join(data_dir, 'ai_models_dataset', 'frontier_ai_models.csv'))
gpu_clusters = pd.read_csv(os.path.join(data_dir, 'gpu_clusters_dataset', 'gpu_clusters.csv'))

print(f"Initial frontier models: {len(frontier_models)}")
print(f"Initial GPU clusters: {len(gpu_clusters)}")

# ============================================================================
# Filter Frontier Models
# ============================================================================
print("\n=== Filtering Frontier Models ===")
print("Keeping only models with known training compute...")

# Filter frontier models to only those with known training compute
frontier_models_filtered = frontier_models[
    frontier_models['Training compute (FLOP)'].notna()
].copy()

print(f"Frontier models with known training compute: {len(frontier_models_filtered)}")

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

# ============================================================================
# Filter GPU Clusters
# ============================================================================
print("\n=== Filtering GPU Clusters ===")
print("Keeping only rank-1 clusters with known 16-bit OP/s...")

# Filter GPU clusters to only rank=1 clusters with known 16-bit OP/s
gpu_clusters_filtered = gpu_clusters[
    (gpu_clusters['Rank when first operational'] == 1) &
    (gpu_clusters['16-bit OP/s (log)'].notna())
].copy()

print(f"Rank-1 GPU clusters with known 16-bit OP/s: {len(gpu_clusters_filtered)}")

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

# ============================================================================
# Calculate Top 10 Models and Training Time for Each Cluster
# ============================================================================
print("\n=== Calculating Training Times ===")
print("For each rank-1 cluster, finding top 10 models released before operational date...")

def get_top_10_models_before_date(models_df, cutoff_date, top_n=10):
    """Get top N models by training compute released before cutoff_date"""
    models_before = models_df[models_df['Publication date'] < cutoff_date]
    if len(models_before) < top_n:
        print(f"Warning: Only {len(models_before)} models found before {cutoff_date}")
        return models_before.nlargest(len(models_before), 'Training compute (FLOP)')
    return models_before.nlargest(top_n, 'Training compute (FLOP)')

# Calculate training time for each cluster based on its top 10 models
results = []

for idx, cluster_row in gpu_clusters_filtered.iterrows():
    cluster_name = cluster_row['Name']
    cluster_date = cluster_row['First Operational Date']
    cluster_ops_log = cluster_row['16-bit OP/s (log)']

    # Get top 10 models before this cluster's operational date
    top_10_for_cluster = get_top_10_models_before_date(frontier_models_filtered, cluster_date)

    if len(top_10_for_cluster) == 0:
        print(f"No models found before {cluster_name} ({cluster_date})")
        continue

    # Calculate average training compute for these models
    avg_compute_log = top_10_for_cluster['Training compute (log)'].mean()
    avg_compute = top_10_for_cluster['Training compute (FLOP)'].mean()

    # Calculate training time
    training_time_log_sec = avg_compute_log - cluster_ops_log
    training_time_sec = 10 ** training_time_log_sec
    training_time_min = training_time_sec / 60
    training_time_hrs = training_time_min / 60
    training_time_days = training_time_hrs / 24

    results.append({
        'Name': cluster_name,
        'Owner': cluster_row['Owner'],
        'First Operational Date': cluster_date,
        '16-bit OP/s (log)': cluster_ops_log,
        'Num models available': len(top_10_for_cluster),
        'Avg training compute (log)': avg_compute_log,
        'Avg training compute (FLOP)': avg_compute,
        'Training time (seconds)': training_time_sec,
        'Training time (minutes)': training_time_min,
        'Training time (hours)': training_time_hrs,
        'Training time (days)': training_time_days
    })

# Create results dataframe
gpu_clusters_results = pd.DataFrame(results)

print(f"\nCalculated training times for {len(gpu_clusters_results)} clusters")

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n" + "=" * 60)
print("Training Time Summary Statistics")
print("=" * 60)
print(f"Median training time: {gpu_clusters_results['Training time (hours)'].median():.2f} hours ({gpu_clusters_results['Training time (days)'].median():.2f} days)")
print(f"Mean training time: {gpu_clusters_results['Training time (hours)'].mean():.2f} hours ({gpu_clusters_results['Training time (days)'].mean():.2f} days)")
print(f"Min training time: {gpu_clusters_results['Training time (hours)'].min():.2f} hours ({gpu_clusters_results['Training time (days)'].min():.2f} days)")
print(f"Max training time: {gpu_clusters_results['Training time (hours)'].max():.2f} hours ({gpu_clusters_results['Training time (days)'].max():.2f} days)")

print("\nDistribution of training times:")
print(gpu_clusters_results[['Training time (hours)', 'Training time (days)', 'Num models available']].describe())

# ============================================================================
# Save Results
# ============================================================================
output_path = os.path.join(data_dir, 'gpu_cluster_training_times.csv')
gpu_clusters_results.to_csv(output_path, index=False)
print(f"\n{'=' * 60}")
print(f"Results saved to {output_path}")
print(f"{'=' * 60}")
