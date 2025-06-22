# %%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
from scipy import stats
import numpy as np

# %%

nf_cdr_filename = './data/NF-CDR.yml'
nf_ccc_filename = './data/NF-CCC.yml'
hrl_ccc_filename = './data/HRL-CCC.yml'
hrl_cdr_filename = './data/HRL-CDR.yml'

# Read all YAML files
with open(nf_cdr_filename, 'r') as file:
    nf_cdr_data = yaml.safe_load(file)
with open(nf_ccc_filename, 'r') as file:
    nf_ccc_data = yaml.safe_load(file)
with open(hrl_ccc_filename, 'r') as file:
    hrl_ccc_data = yaml.safe_load(file)
with open(hrl_cdr_filename, 'r') as file:
    hrl_cdr_data = yaml.safe_load(file)

# Combine all data into a dictionary with labels
# data_dict = {
#     'NF-CDR': nf_cdr_data,
#     'NF-CCC': nf_ccc_data, 
#     'HRL-CCC': hrl_ccc_data,
#     'HRL-CDR': hrl_cdr_data
# }


data_dict = {
    'NF-CDR': nf_cdr_data, 
    'HRL-CDR': hrl_cdr_data
}

# data_dict = {
#     'NF-CCC': nf_ccc_data, 
#     'HRL-CCC': hrl_ccc_data
# }

# data_dict = {
#     'HRL-CDR': hrl_cdr_data,
#     'HRL-CCC': hrl_ccc_data  
# }

# data_dict = {
#     'NF-CCC': nf_ccc_data, 
#     'NF-CDR': nf_cdr_data
# }


# # Choose which data to analyze
# filename = 'HRL-CCC'  # Change this to analyze different datasets
# data = data_dict[filename]


# Prepare data for plotting
all_rows = []
for dataset_name, data in data_dict.items():
    # First, collect reward totals for each reward function to find min/max
    reward_totals = {}
    for result in data['trials']:
        reward_function = result['reward-function']
        if reward_function not in reward_totals:
            reward_totals[reward_function] = []
        for sequence in result['sequences']:
            reward_totals[reward_function].append(sequence['reward-total'])
    
    # Calculate min/max for each reward function
    reward_ranges = {}
    for reward_function, values in reward_totals.items():
        reward_ranges[reward_function] = {
            'min': min(values),
            'max': max(values)
        }
    
    # Now process the data with normalization
    for result in data['trials']:
        reward_function = result['reward-function']
        reward_min = reward_ranges[reward_function]['min']
        reward_max = reward_ranges[reward_function]['max']
        
        for sequence in result['sequences']:
            # Normalize reward_total
            normalized_reward = (sequence['reward-total'] - reward_min) / (reward_max - reward_min) if reward_max != reward_min else 0
            
            all_rows.append({
                'id': result['reward-function'] + '_' + result['sequence-order'] + '_' + str(result['id']),
                'reward_function': result['reward-function'],
                'sequence_order': result['sequence-order'],
                'trial_number': result['id'],
                'state': sequence['state'],
                'sequence_idx': sequence['sequence-idx'],
                'setpoint_avg': sequence['setpoint-avg'],
                'setpoint_dev': sequence['setpoint-dev'],
                'loss_total': sequence['loss-total'],
                'pellets_harvested': sequence['gains-harvested'],
                'gain_total': sequence['gain-total'],
                'reward_total': normalized_reward,  # Use normalized value
                'dataset': dataset_name
            })

# Create DataFrame with all data
df = pd.DataFrame(all_rows)

# Before the plotting code, add t-test analysis
from scipy import stats

def perform_ttest(df, metric):
    datasets = df['dataset'].unique()
    states = df['sequence_idx'].unique()
    
    print(f"\n=== T-Test Results for {metric} ===")
    for state in states:
        print(f"\nState {state}:")
        data1 = df[(df['dataset'] == datasets[0]) & (df['sequence_idx'] == state)][metric]
        data2 = df[(df['dataset'] == datasets[1]) & (df['sequence_idx'] == state)][metric]
        
        # Calculate means and difference
        mean1 = data1.mean()
        mean2 = data2.mean()
        mean_diff = mean1 - mean2
        
        # Calculate Cohen's d
        n1, n2 = len(data1), len(data2)
        var1, var2 = data1.var(), data2.var()
        pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        cohens_d = mean_diff / pooled_se
        
        # Perform t-test
        t_stat, p_val = stats.ttest_ind(data1, data2)
        
        print(f"{datasets[0]} vs {datasets[1]}:")
        print(f"{datasets[0]} mean: {mean1:.4f}")
        print(f"{datasets[1]} mean: {mean2:.4f}")
        print(f"Mean difference: {mean_diff:.4f}")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_val:.4f}")
        print(f"Cohen's d: {cohens_d:.4f}")
        print(f"Significant: {'Yes' if p_val < 0.05 else 'No'}")


# Perform t-tests for each metric
metrics = [
    'setpoint_avg', 'setpoint_dev', 
    'reward_total', 'loss_total',
    'pellets_harvested', 'gain_total'
]

# Update the metrics dictionary for plotting
metrics_dict = {
    'setpoint_avg': 'Setpoint Average',
    'setpoint_dev': 'Setpoint Deviation',
    'reward_total': 'Total Reward Estimate',
    'loss_total': 'Total Loss',
    'gain_total': 'Total Gain',
    'pellets_harvested': 'Pellets Harvested'
}

# Perform t-tests
for metric in metrics:
    perform_ttest(df, metric)

# %%

# Create separate plots for each metric
for metric, title in metrics_dict.items():
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Create box and swarm plots
    sns.boxplot(data=df, x='sequence_idx', y=metric, hue='dataset', ax=ax)
    sns.swarmplot(data=df, x='sequence_idx', y=metric, hue='dataset', 
                 dodge=True, alpha=0.5, size=4, ax=ax)
    
    # Set titles and labels
    ax.set_title(f'{title} by State across Conditions')
    ax.set_xlabel('State')
    ax.set_ylabel(title)
    
    # Update x-axis tick labels
    ax.set_xticklabels(['Canalized', 'Decanalized', 'Recanalized'])
    #ax.set_xticklabels(['Phase 1', 'Phase 2', 'Phase 3'])
    
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Calculate y-axis limits
    y_max = df[metric].max()
    ax.set_ylim(top=y_max * 1.3)
    
    # Adjust legend position
    ax.legend(title='Dataset', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Use tight_layout with adjusted parameters
    plt.tight_layout(rect=(0, 0, 0.95, 1))
    
    # Save plot
    plt.savefig(f'./plots/hrl-nf-cdr-{metric}.png', 
                bbox_inches='tight', dpi=300, pad_inches=0.5)
    plt.close()

# Add after the existing plotting code
# Create a combined plot with all metrics
fig, axes = plt.subplots(3, 2, figsize=(20, 15))
axes = axes.flatten()

# Plot each metric in its own subplot
for idx, (metric, title) in enumerate(metrics_dict.items()):
    # Create box and swarm plots
    sns.boxplot(data=df, x='sequence_idx', y=metric, hue='dataset', ax=axes[idx])
    sns.swarmplot(data=df, x='sequence_idx', y=metric, hue='dataset', 
                 dodge=True, alpha=0.5, size=4, ax=axes[idx])
    
    # Set titles and labels
    axes[idx].set_title(title)
    axes[idx].set_xlabel('State')
    axes[idx].set_ylabel(title)
    
    # Update x-axis tick labels
    axes[idx].set_xticklabels(['Canalized', 'Decanalized', 'Recanalized'])
    #axes[idx].set_xticklabels(['Phase 1', 'Phase 2', 'Phase 3'])
    
    axes[idx].grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Calculate y-axis limits
    y_max = df[metric].max()
    axes[idx].set_ylim(top=y_max * 1.3)
    
    # Adjust legend
    if idx == 0:  # Only keep one legend
        axes[idx].legend(title='Dataset', bbox_to_anchor=(1.02, 1), loc='upper left')
    else:
        axes[idx].get_legend().remove()

# Adjust layout
plt.suptitle('Decanalization by state across HRL and NF', fontsize=16, y=1.02)
plt.tight_layout()

# Save plot
plt.savefig('./plots/all.png', bbox_inches='tight', dpi=300, pad_inches=0.5)
plt.close()



