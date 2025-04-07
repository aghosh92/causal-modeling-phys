Causal Reasoning workflow for learning physics

This repository provides tools for: 
(a) Causal Modeling (Causal Discovery): Discover cause-effect relationships in datasets.
(b) Feature Selection: Select features based on the strength of causal connections, with optional domain-specific context.
(c) Falsifiability Tests: Perform refutability tests to validate the causal model.
(d) Causal Interventions: Conduct targeted interventions for design purposes.
(e) Physical Constraints Evaluation: Evaluate constraints to uncover causal mechanisms in material physics.

## Author

Ayana Ghosh  
Oak Ridge National Laboratory  
Email: research.aghosh@gmail.com  
GitHub: [@aghosh92](https://github.com/aghosh92)

Project Structure
The project is organized as follows:

datasets/           # Folder containing input datasets
outputs/            # Folder for saving output files such as graphs, plots, and results
requirements.txt    # Python package dependencies
scripts/            # Folder containing Python scripts with functions and main execution code
venv/               # Virtual environment for managing project dependencies
Folder Descriptions
datasets/: This folder contains the input datasets for causal analysis. Place all your datasets (e.g., CSV files) in this folder.

outputs/: This folder is used to store all the output files generated during the execution of the scripts. This includes:

Generated plots from refutability tests.

Results such as modified data and model evaluation metrics.

requirements.txt: This file lists the Python dependencies required for the project. To install the necessary libraries, run the following command:

pip install -r requirements.txt
scripts/: This folder contains the Python code for performing causal inference, interventions, and refutability tests. Key files in this folder include:

build_models.py: The main script where causal models are created, intervention strategies are tested, and refutability tests are performed.

functions.py: Contains utility functions for performing causal modeling, visualizations, and refutability tests.

venv/: The virtual environment directory to manage Python dependencies.

Installation
Clone the repository to your local machine:

git clone https://github.com/your-username/causal-modeling-phys.git
Navigate to the project folder:

cd causal-modeling
Create a virtual environment to manage the project's dependencies:

python3 -m venv venv
Activate the virtual environment:

On Windows:

venv\Scripts\activate

On macOS/Linux:

source venv/bin/activate
Install the required dependencies from requirements.txt:

pip install -r requirements.txt
Usage
1. Running the Main Script
You can execute the main script build_models.py to perform causal modeling and refutability tests. This script will:

Load datasets from the datasets/ folder.

Create causal models.

Run interventions (e.g., sequential, independent).

Perform refutability tests (e.g., edge stability, distributional shifts).

Save the results (e.g., graphs, plots) in the outputs/ folder.

To run the script, use the following command:

python scripts/build_models.py
2. Customizing the Analysis
You can modify the build_models.py script to fit your specific dataset and analysis needs:

Change input datasets: Place your CSV datasets in the datasets/ folder and update the code in build_models.py to reference the correct dataset.

Adjust intervention settings: Modify the feats_to_intervene, num_runs, or mode parameters in causal_effect_estimation to control the type of causal interventions you want to test.

Run different refutability tests: The functions.py file includes various refutability tests that one can call from within build_models.py to evaluate your causal model's robustness.

3. Output Files
All output files (e.g., causal model graphs, refutability test results) are saved in the outputs/ folder. The plots for consistency and stability tests will be automatically generated and stored as .png or .pdf files in the outputs/ directory.

Requirements
The project has the following Python dependencies, which are listed in requirements.txt:

numpy (for numerical operations)
pandas (for data manipulation)
matplotlib (for creating plots)
scikit-learn (for machine learning models)
graphviz (for visualizing causal models)

To install the dependencies, use:

pip install -r requirements.txt

