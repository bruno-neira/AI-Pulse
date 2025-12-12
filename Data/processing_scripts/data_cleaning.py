"""
Data Cleaning Script for AI Pulse Project
Author: Michael Chen (mwc2150)

This script performs two main data cleaning operations:
1. frontier_merged: Merges frontiermath benchmark data with frontier_ai_models
2. ai_companies_usage_reports_clean: Converts usage reports to monthly active users
"""

import pandas as pd
import re
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.dirname(script_dir)  # Parent directory (Data/)

print("Starting data cleaning process...")

# ============================================================================
# Part 1: Create frontier_merged.csv
# ============================================================================
print("\n=== Part 1: Creating frontier_merged.csv ===")

# Read input data
frontier_ai_models = pd.read_csv(os.path.join(data_dir, 'ai_models_dataset', 'frontier_ai_models.csv'))
frontiermath = pd.read_csv(os.path.join(data_dir, 'benchmark_dataset', 'frontiermath.csv'))

print(f"Loaded frontier_ai_models: {len(frontier_ai_models)} rows")
print(f"Loaded frontiermath: {len(frontiermath)} rows")

# Define the function to extract a base model name for fuzzy matching
def get_base_model_name_for_search(model_name):
    model_name = str(model_name).lower()

    # General cleanups to get a more accurate base name for matching
    model_name = re.sub(r'\(.*\)', '', model_name).strip() # Remove text in parentheses
    model_name = re.sub(r'mark i', '', model_name).strip() # 'Perceptron Mark I' -> 'perceptron'
    model_name = re.sub(r'nemotron ultra 253b', '', model_name).strip() # 'Llama Nemotron Ultra 253B' -> 'llama'
    model_name = re.sub(r'behemoth', '', model_name).strip() # 'Llama 4 Behemoth' -> 'llama 4'
    model_name = re.sub(r'goliath', '', model_name).strip()
    model_name = re.sub(r'ultra', '', model_name).strip()
    model_name = re.sub(r'nemesis', '', model_name).strip()
    model_name = re.sub(r' \(preview\)', '', model_name).strip()
    model_name = re.sub(r'\s+', '-', model_name).strip() # Replace spaces with hyphens

    # Prioritize specific version extractions for better matching accuracy
    if 'gpt' in model_name:
        match = re.search(r'gpt-?(\d+(?:\.\d+)?)', model_name)
        if match: return 'gpt-' + match.group(1)
        return 'gpt' # Fallback for generic gpt
    if 'grok' in model_name:
        match = re.search(r'grok-?(\d+(?:\.\d+)?)', model_name)
        if match: return 'grok-' + match.group(1)
        return 'grok' # Fallback for generic grok
    if 'llama' in model_name:
        match = re.search(r'llama-?(\d+(?:\.\d+)?)', model_name)
        if match: return 'llama-' + match.group(1)
        return 'llama' # Fallback for generic llama
    if 'claude' in model_name:
        match = re.search(r'claude(?:-?(?:opus|sonnet|haiku))?-?(\d+(?:\.\d+)?)', model_name)
        if match and match.group(1):
            return 'claude-' + match.group(1).replace('-', '.')
        return 'claude' # Fallback for generic claude
    if 'o3' in model_name: return 'o3'
    if 'o1' in model_name: return 'o1'
    if 'o4' in model_name: return 'o4'
    if 'mistral' in model_name: return 'mistral'
    if 'perceptron' in model_name: return 'perceptron'
    if 'theseus' in model_name: return 'theseus'
    if 'pandemonium' in model_name: return 'pandemonium'
    if 'samuel' in model_name: return 'samuel'
    if 'deepseek' in model_name: return 'deepseek'
    if 'qwen' in model_name: return 'qwen'
    if 'gemini' in model_name:
        match = re.search(r'gemini-?(\d+(?:\.\d+)?)', model_name)
        if match: return 'gemini-' + match.group(1)
        return 'gemini'
    if 'kimi' in model_name: return 'kimi'

    return model_name.strip() # Return the cleaned name if no specific pattern matched

# Create cartesian product of the two dataframes
cartesian = frontier_ai_models.assign(key=1).merge(frontiermath.assign(key=1), on='key', suffixes=('_model', '_math')).drop('key', axis=1)

# Add a 'search_term' column based on the 'Model' column
cartesian['search_term'] = cartesian['Model'].apply(get_base_model_name_for_search)

# Filter the cartesian product: check if 'Model version' contains the 'search_term'
mask = cartesian.apply(
    lambda row: isinstance(row['Model version'], str) and
                isinstance(row['search_term'], str) and
                len(row['search_term']) > 2 and
                row['search_term'] in row['Model version'].lower(),
    axis=1
)
merged = cartesian[mask].copy()

# Create the new column 'model_and_version' by concatenating 'Model' and 'Model version'
merged['model_and_version'] = merged['Model'] + ' ' + merged['Model version']

# Drop duplicates based on 'Model version'
merged = merged.drop_duplicates(subset=['Model version'], keep='first')

# Drop the temporary 'search_term' column
merged = merged.drop(columns=['search_term'])

# Save to CSV
output_path = os.path.join(data_dir, 'frontier_merged.csv')
merged.to_csv(output_path, index=False)
print(f"Saved frontier_merged.csv: {len(merged)} rows")

# ============================================================================
# Part 2: Create ai_companies_usage_reports_clean.csv
# ============================================================================
print("\n=== Part 2: Creating ai_companies_usage_reports_clean.csv ===")

# Read input data
df_usage = pd.read_csv(os.path.join(data_dir, 'ai_companies_dataset', 'ai_companies_usage_reports.csv'))
print(f"Loaded ai_companies_usage_reports: {len(df_usage)} rows")

# Define conversion factors for weekly and daily to monthly
# Using average days in a year (365.25) for more accurate monthly conversions
weeks_in_month = (365.25 / 7) / 12  # Approximately 4.348 weeks per month
days_in_month = 365.25 / 12       # Approximately 30.4375 days per month

# Convert 'Weekly' active users to 'Monthly' equivalents
weekly_mask = (df_usage['Active users time period'] == 'Weekly') & (df_usage['Active users'].notna())
df_usage.loc[weekly_mask, 'Active users'] = df_usage.loc[weekly_mask, 'Active users'] * weeks_in_month

# Convert 'Daily' active users to 'Monthly' equivalents
daily_mask = (df_usage['Active users time period'] == 'Daily') & (df_usage['Active users'].notna())
df_usage.loc[daily_mask, 'Active users'] = df_usage.loc[daily_mask, 'Active users'] * days_in_month

# Set all 'Active users time period' entries to 'Monthly'
df_usage['Active users time period'] = 'Monthly'

# Drop rows without active users
df_usage = df_usage.dropna(subset=['Active users'])
print(f"After dropping NaN values: {len(df_usage)} rows")

# Save to CSV
output_path = os.path.join(data_dir, 'ai_companies_usage_reports_clean.csv')
df_usage.to_csv(output_path, index=False)
print(f"Saved ai_companies_usage_reports_clean.csv: {len(df_usage)} rows")

print("\n=== Data cleaning completed successfully! ===")
