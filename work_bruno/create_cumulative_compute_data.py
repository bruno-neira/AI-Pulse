import pandas as pd
import re

# Define the clean_owner_name function
def clean_owner_name(owner):
    if pd.isna(owner):
        return owner

    # Extract organization name from parentheses if present
    paren_match = re.search(r'\(([^)]+)\)', owner)
    if paren_match:
        return paren_match.group(1)

    owner = owner.strip()

    # Handle comma-separated multiple owners
    if ',' in owner:
        owners_list = [o.strip() for o in owner.split(',')]

        ai_first_companies = ['OpenAI', 'Anthropic', 'xAI', 'Meta', 'Google', 'DeepMind',
                              'Alibaba', 'Mistral', 'Cohere', 'Inflection']

        for ai_company in ai_first_companies:
            for o in owners_list:
                if ai_company.lower() in o.lower():
                    owner = o
                    break
            else:
                continue
            break
        else:
            owner = owners_list[0]

    mappings = {
        'Microsoft': 'Microsoft',
        'Meta AI': 'Meta',
        'Meta': 'Meta',
        'Google DeepMind': 'Google DeepMind',
        'DeepMind': 'Google DeepMind',
        'Google': 'Google DeepMind',
        'Amazon': 'Amazon',
        'OpenAI': 'OpenAI',
        'Anthropic': 'Anthropic',
        'Ant Group': 'Alibaba',
        'Alibaba': 'Alibaba',
        'Tesla': 'Tesla',
        'xAI': 'xAI',
        'Oracle': 'Oracle',
        'Tencent': 'Tencent',
        'Mistral': 'Mistral',
        'Cohere': 'Cohere',
        'Inflection': 'Inflection AI'
    }

    for key, value in mappings.items():
        if owner.lower() == key.lower():
            return value

    for key, value in mappings.items():
        if key.lower() in owner.lower():
            return value

    return owner

# Bucket owner function
def bucket_owner(owner):
    if pd.isna(owner):
        return 'Other'

    owner_str = str(owner).lower()

    if 'google' in owner_str or 'deepmind' in owner_str:
        return 'Google DeepMind'
    elif 'openai' in owner_str:
        return 'OpenAI'
    elif 'anthropic' in owner_str:
        return 'Anthropic'
    elif 'xai' in owner_str:
        return 'xAI'
    elif 'meta' in owner_str:
        return 'Meta'
    elif 'alibaba' in owner_str or 'ant group' in owner_str:
        return 'Alibaba'
    elif 'amazon' in owner_str:
        return 'Amazon'
    else:
        return 'Other'

print("Loading datasets...")
# Load GPU clusters
gpu_clusters = pd.read_csv('../Data/gpu_clusters_dataset/gpu_clusters.csv')
notable_models = pd.read_csv('../Data/ai_models_dataset/notable_ai_models.csv')

# Filter to existing clusters with known H100 equivalents
print("Filtering GPU clusters...")
clusters_for_cumulative = gpu_clusters[
    (gpu_clusters['Status'] == 'Existing') &
    (gpu_clusters['H100 equivalents'].notna())
].copy()

print(f"Existing clusters with known H100 equivalents: {len(clusters_for_cumulative)}")

# Clean owner names
clusters_for_cumulative['Owner'] = clusters_for_cumulative['Owner'].apply(clean_owner_name)

# Convert date and compute to proper types
clusters_for_cumulative['First Operational Date'] = pd.to_datetime(
    clusters_for_cumulative['First Operational Date'],
    errors='coerce'
)
clusters_for_cumulative['H100 equivalents'] = pd.to_numeric(
    clusters_for_cumulative['H100 equivalents'],
    errors='coerce'
)

# Drop rows with missing dates
clusters_for_cumulative = clusters_for_cumulative.dropna(subset=['First Operational Date'])
print(f"After cleaning: {len(clusters_for_cumulative)} clusters")

# Calculate cumulative compute for each organization over time
print("\nCalculating cumulative compute...")
clusters_for_cumulative = clusters_for_cumulative.sort_values(['Owner', 'First Operational Date'])
clusters_for_cumulative['cumulative_compute'] = clusters_for_cumulative.groupby('Owner')['H100 equivalents'].cumsum()

# Prepare cluster data for combined dataset
cluster_cumulative_data = clusters_for_cumulative[['Name', 'Owner', 'First Operational Date', 'cumulative_compute']].copy()
cluster_cumulative_data.columns = ['name', 'owner', 'date', 'cumulative_compute']
cluster_cumulative_data['type'] = 'cluster'
cluster_cumulative_data['training_compute'] = pd.NA

# Prepare model data
model_cumulative_data = notable_models[['Model', 'Organization', 'Publication date', 'Training compute (FLOP)']].copy()
model_cumulative_data.columns = ['name', 'owner', 'date', 'training_compute']
model_cumulative_data['cumulative_compute'] = pd.NA
model_cumulative_data['type'] = 'model'
model_cumulative_data['date'] = pd.to_datetime(model_cumulative_data['date'], errors='coerce')
model_cumulative_data['training_compute'] = pd.to_numeric(model_cumulative_data['training_compute'], errors='coerce')

print(f"Cluster data: {len(cluster_cumulative_data)} rows")
print(f"Model data: {len(model_cumulative_data)} rows")

# Apply bucketing
print("\nBucketing organizations...")
cluster_cumulative_data['owner_bucket'] = cluster_cumulative_data['owner'].apply(bucket_owner)
model_cumulative_data['owner_bucket'] = model_cumulative_data['owner'].apply(bucket_owner)

print("\nOwner bucket distribution for clusters:")
print(cluster_cumulative_data['owner_bucket'].value_counts())

# Combine cluster and model data
cumulative_combined = pd.concat([cluster_cumulative_data, model_cumulative_data], ignore_index=True)

print(f"\nCombined dataset: {len(cumulative_combined)} rows")
print(f"Clusters: {len(cumulative_combined[cumulative_combined['type'] == 'cluster'])}")
print(f"Models: {len(cumulative_combined[cumulative_combined['type'] == 'model'])}")

# Save to CSV
cumulative_combined.to_csv('../Data/cumulative_compute_timeline.csv', index=False)
print("\nData saved to ../Data/cumulative_compute_timeline.csv")

print("\n=== Final Summary ===")
print(f"Total rows: {len(cumulative_combined)}")
print(f"\nBy owner bucket:")
print(cumulative_combined.groupby('owner_bucket')['type'].value_counts().unstack(fill_value=0))

# Display sample
print("\nSample of cumulative compute data:")
print(cumulative_combined[cumulative_combined['type'] == 'cluster'].head(10))
