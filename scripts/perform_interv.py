# perform_interv.py

import pandas as pd
import numpy as np
import os
from functions import create_causal_model, visualize_model, causal_effect_estimation 

# Define the path to the datasets
dataset_path = '../datasets/dataset0.csv'
dataset_filtered_path = './outputs/dataset_filtered.csv'
dataset_gp_I_path = '../datasets/df_gp_I.csv'

# Step 1: Read in the datasets
df_filtered = pd.read_csv(dataset_filtered_path)
print(f"Columns in the filtered dataset: {df_filtered.columns}")

df_gp_I = pd.read_csv(dataset_gp_I_path)

df_causal_gp_I = df_gp_I[['TF', 'Adis ', 'rot_angle', 'tilt_angle', 'R-barrier']].copy()
df_causal_gp_I['target'] = df_gp_I['R-barrier']
df_causal_gp_I = df_causal_gp_I.drop(columns=['R-barrier'])  # drop target column
df_causal_gp_I_std = (df_causal_gp_I - df_causal_gp_I.mean()) / df_causal_gp_I.std()

# List of selected features to standardize
selected_features = [
    'TF', 'Adis ', "A'dis ", 'rot_angle', 'tilt_angle',
    'QR+', 'QT', 'QAFEO', 'QJT2D', 'CD2D', 'R-barrier', 'polar_modes', 'Energy'
]

# Standardize the selected features
df_gp_I_std = df_gp_I.copy()  # Make a copy of the original DataFrame
df_gp_I_std[selected_features] = (df_gp_I[selected_features] - 
                                    df_gp_I[selected_features].mean()) / df_gp_I[selected_features].std()

# Step 2: Create the causal model with filtered_dataset
model, adj_matrix, pk = create_causal_model(df_filtered)

#Step 3: Select features to intervene and perform interventions
feats_to_intervene = ['tilt_angle', 'rot_angle']

#perform causal intervention
intervened_X_df_sequential = causal_effect_estimation(df_gp_I_std, df_causal_gp_I_std, 
                                feats_to_intervene, model, mode='sequential',num_runs=100)

percent_change_rot_angle = ((intervened_X_df_sequential['rot_angle'] - \
    df_causal_gp_I_std['rot_angle']) / df_causal_gp_I_std['rot_angle'])*100
percent_change_tilt_angle = ((intervened_X_df_sequential['tilt_angle'] - \
                            df_causal_gp_I_std['tilt_angle']) / df_causal_gp_I_std['tilt_angle'])*100
percent_change_Adis = ((intervened_X_df_sequential['Adis '] - df_causal_gp_I_std['Adis ']) / \
                        df_causal_gp_I_std['Adis '])*100
percent_change_switching = ((intervened_X_df_sequential['intervened_target'] - \
                            df_causal_gp_I_std['target']) / df_causal_gp_I_std['target'])*100

avg_change_angles = (percent_change_rot_angle + percent_change_tilt_angle) / 2

#dataframe percent switching
df_percent_change = pd.DataFrame({
    'compound_name': intervened_X_df_sequential['Compound name'],
    'rot_angle': percent_change_rot_angle,
    'tilt_angle': percent_change_tilt_angle,
    'avg_change_angles': avg_change_angles,
    'Adis': percent_change_Adis,
    'switching': percent_change_switching
})

# Save the data with percentage change in the outputs folder
output_folder = os.path.join(os.getcwd(), 'outputs')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

csv_file_path = os.path.join(output_folder, 'percent_change_data.csv')
df_percent_change.to_csv(csv_file_path, index=False)

print(f"Data saved to: {csv_file_path}")
