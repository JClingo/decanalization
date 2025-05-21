# %%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yaml

# %%

lm_cdr_filename = 'LM-CDR.yml'
lm_ccc_filename = 'LM-CCC.yml'
hrl_ccc_filename = 'HRL-CCC.yml'
hrl_cdr_filename = 'HRL-CDR.yml'

# Read all YAML files
with open(lm_cdr_filename, 'r') as file:
    lm_cdr_data = yaml.safe_load(file)
with open(lm_ccc_filename, 'r') as file:
    lm_ccc_data = yaml.safe_load(file)
with open(hrl_ccc_filename, 'r') as file:
    hrl_ccc_data = yaml.safe_load(file)
with open(hrl_cdr_filename, 'r') as file:
    hrl_cdr_data = yaml.safe_load(file)

# Combine all data into a dictionary with labels
# data_dict = {
#     'LM-CDR': lm_cdr_data,
#     'LM-CCC': lm_ccc_data, 
#     'HRL-CCC': hrl_ccc_data,
#     'HRL-CDR': hrl_cdr_data
# }


data_dict = {
    'LM-CCC': lm_ccc_data, 
    'HRL-CCC': hrl_ccc_data
}

# # Choose which data to analyze
# filename = 'HRL-CCC'  # Change this to analyze different datasets
# data = data_dict[filename]


# Prepare data for plotting
all_rows = []
for dataset_name, data in data_dict.items():
    for result in data['trials']:
        for sequence in result['sequences']:
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
                'gains_harvested': sequence['gains-harvested'],
                'gain_total': sequence['gain-total'],
                'dataset': dataset_name  # Add dataset identifier
            })

# Create DataFrame with all data
df = pd.DataFrame(all_rows)

# %%

fig, (ax1) = plt.subplots(1, 1, figsize=(8, 6))
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# First subplot - Setpoint Averages
sns.boxplot(data=df, x='sequence_idx', y='setpoint_avg', hue='dataset', ax=ax1)
sns.swarmplot(data=df, x='sequence_idx', y='setpoint_avg', hue='dataset', dodge=True, alpha=0.5, size=4, ax=ax1)
ax1.set_title('Setpoint Averages by state across Conditions')
ax1.set_xlabel('State (all 3 are canalizing, just resetting the h value to 0)')
ax1.set_ylabel('Setpoint Average')
ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

# Second subplot - Loss Totals
# sns.boxplot(data=df, x='sequence_idx', y='loss_total', hue='dataset', ax=ax2)
# sns.swarmplot(data=df, x='sequence_idx', y='loss_total', hue='dataset', dodge=True, alpha=0.5, size=4, ax=ax2)
# ax2.set_title('h Loss Totals by State')
# ax2.set_xlabel('State')
# ax2.set_ylabel('Loss Total')
# ax2.grid(True, axis='y', linestyle='--', alpha=0.7)

# Third subplot - Setpoint Deviations
# sns.boxplot(data=df, x='sequence_idx', y='setpoint_dev', hue='dataset', ax=ax3)
# sns.swarmplot(data=df, x='sequence_idx', y='setpoint_dev', hue='dataset', dodge=True, alpha=0.5, size=4, ax=ax3)
# ax3.set_title('Setpoint Deviations by State')
# ax3.set_xlabel('State')
# ax3.set_ylabel('Setpoint Deviation')
# ax3.grid(True, axis='y', linestyle='--', alpha=0.7)

# Adjust layout and legends
plt.tight_layout()
ax1.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
# ax2.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
# ax3.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')

# Save plot
plt.savefig('combined_state_comparisons.png', bbox_inches='tight', dpi=300)
# plt.close()

# %%
