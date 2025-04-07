# build_models.py

import pandas as pd
import numpy as np
import os
from functions import create_causal_model, visualize_model, prune_causal_model, refutability_test_I, refutability_test_II, edge_stability_test, distributional_shift_test, save_and_render_graph

# Define the path to the dataset
dataset_path = '../datasets/dataset0.csv'

# Step 1: Read in the dataset
df = pd.read_csv(dataset_path)
print(f"Columns in the dataset: {df.columns}")

# Prepare a copy of the dataset and add target
df_check = df.copy()
df_check['target'] = df['R-barrier']
df_check = df_check.drop(columns=['R-barrier','Compound name','Energy','rA', 'rA\''])  # Drop target column
df_check['target'] = df['R-barrier']

# Step 2: Standardize the DataFrame (Z-score normalization)
standardized_df_check = (df_check - df_check.mean()) / df_check.std()

# Step 3: Create and visualize the causal model (model I)
model_I, adj_matrix_I, pk_I = create_causal_model(standardized_df_check)
dot = visualize_model(model_I, adj_matrix_I, standardized_df_check, model_name="dag_I")

# Save the adjacency matrix in the outputs folder
output_folder = os.path.join(os.getcwd(), 'outputs')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

adj_matrix_file_path = os.path.join(output_folder, 'adj_matrix_I.npy')
np.save(adj_matrix_file_path, adj_matrix_I)
print(f"Adjacency matrix saved as {adj_matrix_file_path}")

# Step 4: Prune the causal model and visualize (model II)
model_II, adj_matrix_II, discarded_features, dataset_filtered = prune_causal_model(standardized_df_check, threshold_percentile=90)
dot_II = visualize_model(model_II, adj_matrix_II, dataset_filtered, model_name="dag_II")

adj_matrix_file_path = os.path.join(output_folder, 'adj_matrix_II.npy')
np.save(adj_matrix_file_path, adj_matrix_II)
print(f"Adjacency matrix saved as {adj_matrix_file_path}")

# Step 5: Drop specific features and visualize the new causal model (model III)
feature_names_drop_context = ['rad_diff', 'QR+', 'QT', 'QAFEO', 'QJT2D', 'CD2D', 'polar_modes']
dataset_filtered_II = dataset_filtered.drop(
    columns=[col for col in feature_names_drop_context if col in dataset_filtered.columns]
)
print(f"Columns in the dataset after downselection of features: {dataset_filtered_II.columns}")
model_new_II, adj_matrix_new_II, pk_new_II = create_causal_model(dataset_filtered_II)
dot_III = visualize_model(model_new_II, adj_matrix_new_II, dataset_filtered_II, model_name="dag_III")
#save the dataset
csv_file_path = os.path.join(output_folder, 'dataset_filtered.csv')
dataset_filtered_II.to_csv(csv_file_path, index=False)

adj_matrix_file_path = os.path.join(output_folder, 'adj_matrix_new_II.npy')
np.save(adj_matrix_file_path, adj_matrix_new_II)
print(f"Adjacency matrix saved as {adj_matrix_file_path}")

# Step 6: Perform falsifiability tests
selected_features = ['Adis ', 'rot_angle', 'tilt_angle']
statistics = standardized_df_check[selected_features].describe().transpose()[['mean', 'std']].to_dict(orient='index')

# Create a feature statistics dictionary
feature_statistics = {column: {'mean': values['mean'], 'std': values['std']} for column, values in statistics.items()}
for feature, stats in feature_statistics.items():
    data_column = standardized_df_check[feature]
    median_value = np.median(data_column)
    stats['median'] = median_value

# Perform refutability tests
# #make a dictionary of selected features with corresponding mean, median and standard deviations
selected_features = ['Adis ', 'rot_angle','tilt_angle']
# Find mean and standard deviation values in each column
statistics = standardized_df_check[selected_features].describe().transpose()[['mean', 'std']].to_dict(orient='index')
# Convert to the desired format
feature_statistics = {column: {'mean': values['mean'], 'std': values['std']} for column, values in statistics.items()}

for feature, stats in feature_statistics.items():
    data_column = standardized_df_check[feature]
    median_value = np.median(data_column)
    stats['median'] = median_value
#perform falsifiability test
refutability_test_I(dataset_filtered_II, selected_features, feature_statistics, iterations=21)
refutability_test_II(dataset_filtered_II, selected_features, feature_statistics, iterations=21)
distributional_shift_test(dataset_filtered_II, shift_variable='rot_angle', iterations=21)
edge_stability_test(dataset_filtered_II, iterations=21)
