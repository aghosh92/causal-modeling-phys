# functions.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
import lingam
from lingam.utils import make_prior_knowledge
from graphviz import dot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import shutil
import subprocess
import warnings

# Delete the __pycache__ folder if it exists
pycache_dir = "__pycache__"
if os.path.exists(pycache_dir):
    shutil.rmtree(pycache_dir)
    print(f"Deleted {pycache_dir} directory.")
else:
    print(f"{pycache_dir} directory does not exist.")

# Get the directory where the current script is located
script_path = os.path.realpath(__file__)
script_dir = os.path.dirname(script_path)

# Go one level up from the script directory — this should be where the 'scripts' folder sits
base_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

# Output folder in the same directory as 'scripts'
output_folder = os.path.join(base_dir, 'outputs')

# Ensure it exists
os.makedirs(output_folder, exist_ok=True)


def create_causal_model(data, exogenous_variables=None, sink_variables=None, paths=None, no_paths=None):
    """
    Creates a causal model using the provided dataset.

    Parameters:
    - data (pd.DataFrame): The dataset to use for creating the causal model.
    - exogenous_variables (list): Optional list of exogenous variables.
    - sink_variables (list): Optional list of sink variables.
    - paths (list): Optional list of causal paths to be considered.
    - no_paths (list): Optional list of paths to exclude.

    Returns:
    - model (lingam.DirectLiNGAM): The fitted causal model.
    - adj_matrix (np.array): The adjacency matrix representing the causal relationships.
    - pk (lingam.utils.PriorKnowledge): The prior knowledge used in the model.
    """
    # Create prior knowledge for the model
    pk = make_prior_knowledge(n_variables=len(data.columns), sink_variables=[data.shape[1] - 1])
    
    # Instantiate and fit the causal model
    model = lingam.DirectLiNGAM(prior_knowledge=pk)
    model.fit(data)
    
    # Get the adjacency matrix (shows the causal relationships between variables)
    adj_matrix = model.adjacency_matrix_

    return model, adj_matrix, pk


def visualize_model(model, adj_matrix, data, model_name="causal_model"):
    """
    Visualizes the causal model using Graphviz and saves the output as a .pdf.

    Parameters:
    - model (lingam.DirectLiNGAM): The fitted causal model.
    - adj_matrix (np.array): The adjacency matrix of the model.
    - data (pd.DataFrame): The dataset to use for naming the variables.
    - model_name (str): The name for saving the output graph (e.g., 'dag_I').

    Returns:
    - dot_graph (graphviz.Digraph): The generated Graphviz Digraph object.
    """
    # Set a threshold to consider connections stronger than 0.01
    idx = np.abs(adj_matrix) > 0.01
    dirs = np.where(idx)

    # Create the Graphviz Digraph object
    dot_graph = graphviz.Digraph(engine='dot')

    # Label the variables in the graph
    labels = [f'{i}. {col}' for i, col in enumerate(data.columns)]
    names = labels if labels else [f'x{i}' for i in range(len(adj_matrix))]

    # Add edges to the graph based on the adjacency matrix
    for to, from_, coef in zip(dirs[0], dirs[1], adj_matrix[idx]):
        dot_graph.edge(names[from_], names[to], label=f'{coef:.2f}')

    return dot_graph

def prune_causal_model(dataset, threshold_percentile=90):
    """
    Processes the dataset by creating a causal model, discarding weak features, and visualizing the result.

    Parameters:
    - dataset (pd.DataFrame): The dataset for the causal model.
    - threshold_percentile (int): Percentile threshold for discarding weak features (default is 90).

    Returns:
    - model_new (lingam.DirectLiNGAM): The new causal model after pruning.
    - adj_matrix_new (np.array): The updated adjacency matrix after pruning.
    - discarded_features (list): List of features that were discarded due to weak connections.
    - dataset_filtered (pd.DataFrame): The dataset after discarding the weak features.
    """
    # Create the initial causal model using the entire dataset
    model_all, adj_matrix_all, pk_all = create_causal_model(dataset)

    # Determine the threshold based on the given percentile of the absolute values in the adjacency matrix
    threshold = np.percentile(np.abs(adj_matrix_all), threshold_percentile)
    adj_matrix_all[np.abs(adj_matrix_all) < threshold] = 0

    # Identify discarded features (those with weak connections)
    discarded_features = []
    feature_names = dataset.columns.tolist()

    # Loop through each feature and check if its sum of absolute values in the adjacency matrix is below the threshold
    for i, feature_name in enumerate(feature_names):
        if np.sum(np.abs(adj_matrix_all[i])) == 0:  # If no connections remain for this feature
            discarded_features.append(feature_name)

    # Print the discarded features
    if discarded_features:
        print(f"Discarded features (weak connections based on the threshold): {discarded_features}")
    else:
        print("No features discarded based on the threshold.")

    # Create the new dataset after discarding the features
    dataset_filtered = dataset.drop(columns=discarded_features)

    # Create the new causal model with the filtered dataset
    model_new, adj_matrix_new, pk_new = create_causal_model(dataset_filtered)

    return model_new, adj_matrix_new, discarded_features, dataset_filtered

# Ensure the outputs folder exists
if not os.path.exists('outputs'):
    os.makedirs('outputs')

def refutability_test_I(data, features, feature_statistics, iterations=10):
    """
    Refutability Test based on causal model structure by comparing the adjacency matrix values.
    
    Parameters:
    - data: The dataset to test.
    - features: List of features to apply refutability constraints.
    - feature_statistics: Statistics (mean, std) for the features to apply constraints.
    - iterations: Number of iterations to perform the test.
    """
    print('Refutability Test I: Based on causal model structure')
    
    consistent_counts = []  # To store the count of consistent results
    inconsistent_counts = []  # To store the count of inconsistent results
    
    falsification_data = data.copy()  # Create a copy of the data to manipulate

    # Run multiple iterations
    for j in range(iterations):
        for feature in features:
            # Apply feature constraints
            if feature in feature_statistics:
                constraint = feature_statistics[feature]
                mean_value = constraint['mean']
                std_value = constraint['std']

                # Apply random perturbation to the feature based on its statistics
                diff_from_mean = np.abs(data[feature] - mean_value)
                num = np.abs(np.random.uniform(low=-diff_from_mean, high=diff_from_mean) * 0.15)
                falsification_data[feature] = data[feature] + num

        # Build causal model on the falsified data
        falsification_model, adj_mat_falsi, _ = create_causal_model(falsification_data)
        # Build causal model on the original data
        model, adj_matrix, pk = create_causal_model(data)

        # Check if the adjacency matrices are similar with a tolerance value
        if np.allclose(model.adjacency_matrix_, adj_mat_falsi, atol=1e-1):
            consistent_counts.append(consistent_counts[-1] + 1 if consistent_counts else 1)
            inconsistent_counts.append(inconsistent_counts[-1] if inconsistent_counts else 0)
        else:
            consistent_counts.append(consistent_counts[-1] if consistent_counts else 0)
            inconsistent_counts.append(inconsistent_counts[-1] + 1 if inconsistent_counts else 1)

    # Plot the results
    iterations_range = range(1, iterations + 1)
    plt.bar(iterations_range, consistent_counts, color='green', label='Consistent')
    plt.bar(iterations_range, inconsistent_counts, color='red', bottom=consistent_counts, label='Inconsistent')
    plt.xlabel('Iterations')
    plt.ylabel('Count')
    plt.title('Consistency of Causal Model Structure Across Iterations')
    plt.legend()

    # Save the plot to the outputs folder
    plt.savefig('outputs/falsifiability_test_I.png')
    plt.close()  # Close the plot to avoid display in interactive environments

def refutability_test_II(data, features, feature_statistics, iterations=10):
    """
    Refutability Test based on causal model order by checking if the causal ordering remains consistent.
    
    Parameters:
    - data: The dataset to test.
    - features: List of features to apply refutability constraints.
    - feature_statistics: Statistics (mean, std) for the features to apply constraints.
    - iterations: Number of iterations to perform the test.
    """
    print('Refutability Test II: Based on causal model order')
    
    consistent_counts = []  # To store the count of consistent results
    inconsistent_counts = []  # To store the count of inconsistent results

    falsification_data = data.copy()  # Create a copy of the data to manipulate
    matrix_size = len(data.columns)  # Get the size of the data matrix

    # Run multiple iterations
    for j in range(iterations):
        for feature in features:
            # Apply feature constraints
            if feature in feature_statistics:
                constraint = feature_statistics[feature]
                mean_value = constraint['mean']
                std_value = constraint['std']

                # Apply random perturbation to the feature based on its statistics
                diff_from_mean = np.abs(data[feature] - mean_value)
                num = np.abs(np.random.uniform(low=-diff_from_mean, high=diff_from_mean) * 0.1)
                falsification_data[feature] = data[feature] + num

        # Build causal model on the falsified data
        falsification_model, adj_mat_falsi, _ = create_causal_model(falsification_data)
        # Build causal model on the original data
        model, adj_matrix, pk = create_causal_model(data)

        # Extract the adjacency matrix from the falsified model
        falsification_adj_matrix = falsification_model.adjacency_matrix_[:matrix_size, :matrix_size]

        # Check if the adjacency matrices are similar with a tolerance value
        if np.allclose(adj_matrix, falsification_adj_matrix, atol=1e-1):
            consistent_counts.append(consistent_counts[-1] + 1 if consistent_counts else 1)
            inconsistent_counts.append(inconsistent_counts[-1] if inconsistent_counts else 0)
        else:
            consistent_counts.append(consistent_counts[-1] if consistent_counts else 0)
            inconsistent_counts.append(inconsistent_counts[-1] + 1 if inconsistent_counts else 1)

    # Plot the results
    iterations_range = range(1, iterations + 1)
    plt.bar(iterations_range, consistent_counts, color='green', label='Consistent')
    plt.bar(iterations_range, inconsistent_counts, color='red', bottom=consistent_counts, label='Inconsistent')
    plt.xlabel('Iterations')
    plt.ylabel('Count')
    plt.title('Order Consistency of Causal Model Structure Across Iterations')
    plt.legend()

    # Save the plot to the outputs folder
    plt.savefig('outputs/falsifiability_test_II.png')
    plt.close()  # Close the plot to avoid display in interactive environments


def edge_stability_test(data, iterations=10):
    """
    Refutability Test based on edge stability by checking if causal edges remain consistent.
    
    Parameters:
    - data: The dataset to test.
    - iterations: Number of iterations to perform the test.
    """
    print('Refutability Test: Based on Edge Stability')

    consistent_counts = []  # To store the count of consistent results
    inconsistent_counts = []  # To store the count of inconsistent results

    # Run multiple iterations
    for j in range(iterations):
        # Add small random noise to the data to perturb it
        perturbed_data = data + np.random.normal(size=data.shape) * 1e-2

        # Build causal model on the perturbed data
        perturbed_model, adj_mat_pert, _ = create_causal_model(perturbed_data)
        # Build causal model on the original data
        model, adj_matrix, pk = create_causal_model(data)

        # Check if the adjacency matrices are similar with a tolerance value
        if np.allclose(adj_matrix, adj_mat_pert, atol=1e-2):
            consistent_counts.append(consistent_counts[-1] + 1 if consistent_counts else 1)
            inconsistent_counts.append(inconsistent_counts[-1] if inconsistent_counts else 0)
        else:
            consistent_counts.append(consistent_counts[-1] if consistent_counts else 0)
            inconsistent_counts.append(inconsistent_counts[-1] + 1 if inconsistent_counts else 1)

    # Plot the results
    iterations_range = range(1, iterations + 1)
    plt.bar(iterations_range, consistent_counts, color='green', label='Consistent')
    plt.bar(iterations_range, inconsistent_counts, color='red', bottom=consistent_counts, label='Inconsistent')
    plt.xlabel('Iterations')
    plt.ylabel('Count')
    plt.title('Edge Stability Test on Causal Models Across Iterations')
    plt.legend()

    # Save the plot to the outputs folder
    plt.savefig('outputs/edge_stability_test.png')
    plt.close()  # Close the plot to avoid display in interactive environments


def distributional_shift_test(data, shift_variable, iterations=10):
    """
    Refutability Test based on distributional shift by checking if a variable's distribution changes.
    
    Parameters:
    - data: The dataset to test.
    - shift_variable: The variable whose distribution is to be shifted.
    - iterations: Number of iterations to perform the test.
    """
    print('Refutability Test: Based on Distributional Shift')

    consistent_counts = []  # To store the count of consistent results
    inconsistent_counts = []  # To store the count of inconsistent results

    # Run multiple iterations
    for j in range(iterations):
        # Perturb the shift variable by adding small random noise
        shifted_data = data.copy()
        shifted_data[shift_variable] += np.random.normal(size=len(data)) * 1e-5

        # Build causal model on the shifted data
        shifted_model, adj_mat_shift, _ = create_causal_model(shifted_data)
        # Build causal model on the original data
        model, adj_matrix, pk = create_causal_model(data)

        # Check if the adjacency matrices are similar with a tolerance value
        if np.allclose(adj_matrix, adj_mat_shift):
            consistent_counts.append(consistent_counts[-1] + 1 if consistent_counts else 1)
            inconsistent_counts.append(inconsistent_counts[-1] if inconsistent_counts else 0)
        else:
            consistent_counts.append(consistent_counts[-1] if consistent_counts else 0)
            inconsistent_counts.append(inconsistent_counts[-1] + 1 if inconsistent_counts else 1)

    # Plot the results
    iterations_range = range(1, iterations + 1)
    plt.bar(iterations_range, consistent_counts, color='green', label='Consistent')
    plt.bar(iterations_range, inconsistent_counts, color='red', bottom=consistent_counts, label='Inconsistent')
    plt.xlabel('Iterations')
    plt.ylabel('Count')
    plt.title('Distributional Shift Test on Causal Models Across Iterations')
    plt.legend()

    # Save the plot to the outputs folder
    plt.savefig('outputs/distributional_shift_test.png')
    plt.close()  # Close the plot to avoid display in interactive environments

def causal_effect_estimation(df_gp, df_causal_gp, feats_to_intervene, model, mode, num_runs, target_value=None):
    """
    Perform causal intervention on specified features and average the results over multiple runs.

    Parameters:
    - df_gp: Original DataFrame (for tracking compounds).
    - df_causal_gp: Processed DataFrame used for causal inference.
    - feats_to_intervene: List of features to intervene on.
    - mode: 'joint' (intervene on all features together), 
            'independent' (one at a time), 
            'sequential' (step-by-step).
    - target_value: Fixed target value to achieve (if None, sample within distribution).
    - num_runs: Number of runs to perform for averaging the interventions.

    Returns:
    - DataFrame with averaged interventions applied over multiple runs.
    """

    # Get column names and split the data into features (X) and target (y)
    column_names = df_causal_gp.columns
    X = df_causal_gp.iloc[:, :-1].values  # Convert to numpy array for fast updates
    y = df_causal_gp.iloc[:, -1].values

    # Fit linear regression model to the data (for causal effect estimation)
    lr = LinearRegression()
    regressor = lr.fit(X, y)

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)

    # Initialize the causal effect estimation model
    ce = lingam.CausalEffect(model)

    # Get indices of the features to intervene on
    feat_indices = [column_names.get_loc(feat) for feat in feats_to_intervene if feat in column_names]

    # Initialize lists to store the results of each run
    all_intervened_X = []  # To store the feature matrix after intervention for each run
    all_intervened_target_values = []  # To store the target values after intervention for each run

    # Loop over the number of runs for averaging
    for run in range(num_runs):
        np.random.seed(run)  # Ensure variability across runs

        # Start with a fresh copy of the data for each run
        intervened_X = np.copy(X)
        compound_names = []
        intervened_target_values = []

        # Loop through each sample in the dataset
        for k in range(X.shape[0]):
            # Choose target value dynamically or use the given one
            if target_value is not None:
                target_value_k = target_value
            else:
                original_target_value = df_causal_gp['target'].iloc[k]
                target_value_k = original_target_value * np.random.uniform(0.5, 1)  # Target decrease 0% - 50%

            # Sequential intervention mode: Apply interventions step-by-step
            if mode == 'sequential':
                # Randomize the order of features to intervene on to avoid fixed patterns
                np.random.shuffle(feat_indices)
                for feat_index in feat_indices:
                    original_value = intervened_X[k, feat_index]  # Use updated value after previous interventions
                    random_change = np.random.uniform(-0.35, 0.35)
                    # Calculate the valid range for the feature based on min/max values in the original data
                    min_value = max(original_value + random_change * abs(original_value), df_causal_gp[column_names[feat_index]].min())
                    max_value = min(original_value + random_change * abs(original_value), df_causal_gp[column_names[feat_index]].max())

                    # Estimate optimal intervention using causal effect model
                    intervened_value = ce.estimate_optimal_intervention(
                        df_causal_gp.iloc[k, :].values.reshape(1, -1), 4, regressor, feat_index, target_value_k
                    )

                    # Calculate the magnitude of the required change
                    change_magnitude = abs(intervened_value - original_value)

                    # Define threshold for small changes (adjust as needed)
                    threshold = 0.6  # Example threshold, modify based on data

                    # Adaptive weighting: Smaller changes give more weight to original, larger changes give more weight to intervened value
                    weight_original = max(0, 1 - change_magnitude / threshold)
                    weight_intervened = 1 - weight_original

                    # Apply weighted update to the feature
                    intervened_X[k, feat_index] = weight_original * original_value + weight_intervened * np.clip(intervened_value, min_value, max_value)

            # Store the target value after intervention
            intervened_target_values.append(target_value_k)
            # Track the compound name for this sample
            compound_names.append(df_gp.iloc[k]['Compound name'])

        # Store the results of this run
        all_intervened_X.append(intervened_X)
        all_intervened_target_values.append(intervened_target_values)

    # Convert the list of arrays into numpy arrays for averaging
    all_intervened_X = np.array(all_intervened_X)
    all_intervened_target_values = np.array(all_intervened_target_values)

    # Average the results over all runs
    averaged_intervened_X = np.mean(all_intervened_X, axis=0)
    averaged_intervened_target_values = np.mean(all_intervened_target_values, axis=0)

    # Convert the averaged result to a DataFrame
    averaged_intervened_X_df = pd.DataFrame(averaged_intervened_X, columns=column_names[:-1])
    averaged_intervened_X_df['Compound name'] = compound_names
    averaged_intervened_X_df['intervened_target'] = averaged_intervened_target_values

    return averaged_intervened_X_df

def save_and_render_graph(dot, model_name="causal_model", output_folder='outputs'):
    """
    Saves and renders the causal model as a .pdf using Graphviz.

    Parameters:
    - dot_graph (graphviz.Digraph): The Graphviz Digraph object.
    - model_name (str): The name for saving the output graph (e.g., 'dag_I').
    - output_folder (str): The folder where the output graph should be saved.

    Returns:
    - None
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the .dot file temporarily
    dot_file_path = os.path.join(output_folder, f'{model_name}.dot')
    dot.save(dot_file_path)  # Save the .dot file

    # Convert .dot to .pdf using subprocess (assuming 'dot' is installed)
    pdf_file_path = os.path.join(output_folder, f'{model_name}.pdf')

    # Run the Graphviz 'dot' command using subprocess to generate the .pdf
    subprocess.run(['dot', '-Tpdf', dot_file_path, '-o', pdf_file_path], check=True)

    # Print the result
    print(f"Graph saved at: {pdf_file_path}")
