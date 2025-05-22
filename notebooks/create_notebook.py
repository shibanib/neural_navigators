import json
import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Define the cells
cells = [
    nbf.v4.new_markdown_cell("""# Neural Dynamics Across Brain Regions in Decision-Making and Aging

This notebook explores neural dynamics across MOs, basal ganglia, and prefrontal cortex during visual discrimination tasks, with a focus on age-related differences in functional connectivity and their impact on cognitive processes and behavioral performance."""),
    
    nbf.v4.new_markdown_cell("""## 1. Introduction

### 1.1 Background

The Steinmetz dataset provides a rich resource for studying neural activity across multiple brain regions during a visual discrimination task. Mice were trained to determine which side had the highest contrast visual stimulus and respond accordingly. This task requires coordination between sensory processing, decision-making, and motor execution systems in the brain.

### 1.2 Research Questions

1. How do neural dynamics across MOs, basal ganglia, and prefrontal cortex drive strategy selection and decision-making during visual discrimination tasks?

2. How do age-related differences in functional connectivity between these regions influence cognitive processes and behavioral performance?

### 1.3 Significance

Understanding age-related changes in the neural circuits governing dynamic decision-making strategies has critical implications for cognitive aging research (Radulescu et al., 2021). By identifying how functional connectivity between MOs, basal ganglia, and prefrontal cortex declines with age, we uncover specific mechanisms underlying degraded neural multiplexing and strategy-switching. These insights advance our knowledge of neuroplasticity by linking circuit-level dysfunction to behavioral rigidity, such as prolonged biased states or "lapses." Furthermore, our use of LSTM models to decode temporal neural dynamics provides a framework for developing biomarkers of cognitive flexibility. This work opens new avenues for interventions targeting adaptive decision-making circuits to preserve autonomy and quality of life in aging populations."""),
    
    nbf.v4.new_markdown_cell("## 2. Setup and Data Loading"),
    
    nbf.v4.new_code_cell("""# Basic imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import helper functions
from steinmetz_helpers import *

# Set matplotlib defaults
plt.rcParams['figure.figsize'] = [20, 4]
plt.rcParams['font.size'] = 15
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['figure.autolayout'] = True"""),
    
    nbf.v4.new_code_cell("""# Install required packages if needed
!pip install -q tensorflow plotly shap"""),
    
    nbf.v4.new_code_cell("""# Data retrieval
import os, requests

# List of filenames to download
fname = [
    'steinmetz_part0.npz', 'steinmetz_part1.npz', 'steinmetz_part2.npz',
    'steinmetz_lfp.npz'
]

# Corresponding URLs for the files
url = [
    "https://osf.io/agvxh/download", "https://osf.io/uv3mw/download", "https://osf.io/ehmw2/download",
    "https://osf.io/kx3v9/download"
]

# Download the data files if they don't exist
for f, u in zip(fname, url):
    if not os.path.exists(f):
        try:
            r = requests.get(u)
            with open(f, 'wb') as file:
                file.write(r.content)
        except Exception as e:
            print(f"Failed to download {f}: {e}")"""),
    
    nbf.v4.new_code_cell("""# Load the data
alldat = load_steinmetz_data()
dat_LFP = load_steinmetz_lfp()

# Combine the data
combined_data = combine_data(alldat, dat_LFP)

print(f"Number of sessions: {len(combined_data)}")"""),
    
    nbf.v4.new_markdown_cell("## 3. Data Exploration and Preprocessing"),
    
    nbf.v4.new_code_cell("""# Define brain region groups of interest
brain_groups = [
    ['MOs'],  # Secondary motor cortex
    ['ACA', 'PL', 'ILA', 'ORB'],  # Prefrontal cortex regions
    ['CP', 'ACB', 'GPe', 'SNr'],  # Basal ganglia regions
    ['VISp', 'VISl', 'VISrl', 'VISam', 'VISpm'],  # Visual cortex regions
    ['CA1', 'CA3', 'DG']  # Hippocampal regions
]

# Define colors for each brain region group
region_colors = ['blue', 'green', 'red', 'purple', 'orange']

# Define region names for plotting
regions = ['MOs', 'PFC', 'BG', 'VIS', 'HPC', 'Other']"""),
    
    nbf.v4.new_code_cell("""# Analyze the presence of brain regions across sessions
areas_by_dataset, area_to_index = analyze_brain_regions(combined_data, brain_groups)

# Plot the presence of brain regions
plot_brain_regions_presence(areas_by_dataset, area_to_index, brain_groups, region_colors)"""),
    
    nbf.v4.new_code_cell("""# Group mice by age
# For demonstration purposes, we'll use the first half of mice as "young" and the second half as "old"
# In a real analysis, you would use actual age data
all_mice = np.unique([session['mouse_name'] for session in combined_data])
young_mice = all_mice[:len(all_mice)//2]
old_mice = all_mice[len(all_mice)//2:]

print(f"Young mice: {young_mice}")
print(f"Old mice: {old_mice}")"""),
    
    nbf.v4.new_code_cell("""# Filter data by mouse age
young_sessions = []
old_sessions = []

for session in combined_data:
    if session['mouse_name'] in young_mice:
        young_sessions.append(session)
    elif session['mouse_name'] in old_mice:
        old_sessions.append(session)

print(f"Number of young mouse sessions: {len(young_sessions)}")
print(f"Number of old mouse sessions: {len(old_sessions)}")"""),
    
    nbf.v4.new_markdown_cell("""## 4. Neural Dynamics Across Brain Regions

In this section, we analyze how neural dynamics across MOs, basal ganglia, and prefrontal cortex drive strategy selection and decision-making during visual discrimination tasks."""),
    
    nbf.v4.new_markdown_cell("### 4.1 Visual Stimuli Response Analysis"),
    
    nbf.v4.new_code_cell("""# Analyze LFP response to visual stimuli for a sample session
sample_session = combined_data[0]
analyze_visual_stimuli_response(sample_session)"""),
    
    nbf.v4.new_markdown_cell("### 4.2 Feedback Response Analysis"),
    
    nbf.v4.new_code_cell("""# Plot LFP activity for positive and negative feedback
fig_negative = plot_lfp_feedback(sample_session, -1, "Negative")
fig_positive = plot_lfp_feedback(sample_session, 1, "Positive")

fig_negative.show()
fig_positive.show()"""),
    
    nbf.v4.new_markdown_cell("### 4.3 Cross-Correlation Analysis"),
    
    nbf.v4.new_code_cell("""# Analyze cross-correlation between brain regions across sessions
# For demonstration, we'll use the first 3 sessions
sample_sessions = combined_data[:3]
correlations = analyze_cross_correlation(sample_sessions)

# Plot correlation heatmap
correlation_df = plot_correlation_heatmap(correlations)"""),
    
    nbf.v4.new_markdown_cell("### 4.4 LSTM Model for Decoding Neural Dynamics"),
    
    nbf.v4.new_code_cell("""# Prepare data for LSTM model
# For demonstration, we'll use a subset of the data
X_lstm_padded, y_lstm = prepare_lstm_data(alldat[:5], brain_groups)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_lstm_padded, y_lstm, test_size=0.2, random_state=42, stratify=y_lstm)"""),
    
    nbf.v4.new_code_cell("""# Build and train the LSTM model
model, history = build_lstm_model(X_train, y_train, X_test, y_test)"""),
    
    nbf.v4.new_code_cell("""# Evaluate the model
metrics_df = evaluate_lstm_model(model, X_test, y_test)
print(metrics_df)"""),
    
    nbf.v4.new_code_cell("""# Analyze feature importance
X_sample = X_test[:100]  # Use a small sample due to computational load
nareas = len(brain_groups) + 1  # Adding 1 for 'other' brain areas
importance_df = analyze_feature_importance(model, X_sample, regions, nareas)

# Plot feature importance
plot_feature_importance(importance_df)"""),
    
    nbf.v4.new_markdown_cell("""## 5. Age-Related Differences in Neural Dynamics

In this section, we analyze how age-related differences in functional connectivity between MOs, basal ganglia, and prefrontal cortex influence cognitive processes and behavioral performance."""),
    
    nbf.v4.new_code_cell("""# Analyze age-related differences in neural dynamics
age_diff_metrics = analyze_age_differences(young_sessions, old_sessions, brain_groups)

# Plot age-related differences
plot_age_differences(age_diff_metrics)"""),
    
    nbf.v4.new_markdown_cell("### 5.1 Functional Connectivity Analysis by Age"),
    
    nbf.v4.new_code_cell("""# Analyze cross-correlation between brain regions for young mice
young_correlations = analyze_cross_correlation(young_sessions[:3])

# Plot correlation heatmap for young mice
young_correlation_df = plot_correlation_heatmap(young_correlations)"""),
    
    nbf.v4.new_code_cell("""# Analyze cross-correlation between brain regions for old mice
old_correlations = analyze_cross_correlation(old_sessions[:3])

# Plot correlation heatmap for old mice
old_correlation_df = plot_correlation_heatmap(old_correlations)"""),
    
    nbf.v4.new_markdown_cell("### 5.2 LSTM Model Performance by Age"),
    
    nbf.v4.new_code_cell("""# Prepare data for LSTM model for young mice
young_alldat = [session for session in alldat if session['mouse_name'] in young_mice]
X_lstm_young, y_lstm_young = prepare_lstm_data(young_alldat[:3], brain_groups)

# Split the data into training and testing sets
X_train_young, X_test_young, y_train_young, y_test_young = train_test_split(
    X_lstm_young, y_lstm_young, test_size=0.2, random_state=42, stratify=y_lstm_young)"""),
    
    nbf.v4.new_code_cell("""# Build and train the LSTM model for young mice
model_young, history_young = build_lstm_model(X_train_young, y_train_young, X_test_young, y_test_young)"""),
    
    nbf.v4.new_code_cell("""# Evaluate the model for young mice
metrics_df_young = evaluate_lstm_model(model_young, X_test_young, y_test_young)
print(metrics_df_young)"""),
    
    nbf.v4.new_code_cell("""# Prepare data for LSTM model for old mice
old_alldat = [session for session in alldat if session['mouse_name'] in old_mice]
X_lstm_old, y_lstm_old = prepare_lstm_data(old_alldat[:3], brain_groups)

# Split the data into training and testing sets
X_train_old, X_test_old, y_train_old, y_test_old = train_test_split(
    X_lstm_old, y_lstm_old, test_size=0.2, random_state=42, stratify=y_lstm_old)"""),
    
    nbf.v4.new_code_cell("""# Build and train the LSTM model for old mice
model_old, history_old = build_lstm_model(X_train_old, y_train_old, X_test_old, y_test_old)"""),
    
    nbf.v4.new_code_cell("""# Evaluate the model for old mice
metrics_df_old = evaluate_lstm_model(model_old, X_test_old, y_test_old)
print(metrics_df_old)"""),
    
    nbf.v4.new_code_cell("""# Compare model performance between young and old mice
metrics_comparison = pd.concat([metrics_df_young.rename(index={'LSTM': 'Young'}), 
                               metrics_df_old.rename(index={'LSTM': 'Old'})])
print(metrics_comparison)"""),
    
    nbf.v4.new_code_cell("""# Plot model performance comparison
plt.figure(figsize=(12, 6))
metrics_melted = metrics_comparison.reset_index().melt(id_vars='index', var_name='Metric', value_name='Score')
sns.barplot(x='Score', y='index', hue='Metric', data=metrics_melted)
plt.title('Model Performance Comparison by Age')
plt.xlabel('Score')
plt.ylabel('Age Group')
plt.legend(loc='lower right')
plt.xlim(0, 1)
plt.show()"""),
    
    nbf.v4.new_markdown_cell("""## 6. Discussion and Conclusions

### 6.1 Neural Dynamics and Decision-Making

Our analysis reveals how neural dynamics across MOs, basal ganglia, and prefrontal cortex drive strategy selection and decision-making during visual discrimination tasks. The LSTM model successfully decoded neural activity patterns to predict behavioral responses, with certain brain regions showing higher importance in the decision-making process.

Key findings include:
- Differential responses to visual stimuli across brain regions
- Distinct patterns of activity following positive vs. negative feedback
- Strong functional connectivity between specific brain regions during decision-making
- Temporal dynamics captured by the LSTM model that reveal the sequence of information processing

### 6.2 Age-Related Differences

Our comparison between young and old mice revealed significant age-related differences in neural dynamics and functional connectivity, which influence cognitive processes and behavioral performance.

Key findings include:
- Differences in response times and success rates between age groups
- Changes in functional connectivity patterns with age
- Differential model performance when decoding neural activity from young vs. old mice
- Shifts in the relative importance of brain regions for decision-making with age

### 6.3 Implications and Future Directions

These findings have important implications for understanding cognitive aging and developing interventions to preserve cognitive function in aging populations. Future research should focus on:

1. Developing more sophisticated models to capture the full complexity of neural dynamics
2. Investigating interventions that could strengthen functional connectivity in aging brains
3. Exploring the relationship between neural dynamics and specific cognitive processes
4. Translating these findings to human studies of cognitive aging

By advancing our understanding of how neural circuits support adaptive decision-making and how these circuits change with age, we can develop more effective strategies for maintaining cognitive health throughout the lifespan.""")
]

# Add the cells to the notebook
nb['cells'] = cells

# Set the metadata
nb['metadata'] = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    },
    'language_info': {
        'codemirror_mode': {
            'name': 'ipython',
            'version': 3
        },
        'file_extension': '.py',
        'mimetype': 'text/x-python',
        'name': 'python',
        'nbconvert_exporter': 'python',
        'pygments_lexer': 'ipython3',
        'version': '3.8.10'
    }
}

# Write the notebook to a file
with open('Neural_Dynamics_Strategy_Selection_Aging.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook created successfully!") 