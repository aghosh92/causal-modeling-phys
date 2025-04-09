## Causal Reasoning workflow for learning physics
This repository provides tools for:

    --Causal Modeling (Causal Discovery): Discover cause-effect relationships in datasets,
    --Feature Selection: Select features based on the strength of causal connections, with optional domain-specific context,
    --Falsifiability Tests: Perform refutability tests to validate the causal model,
    --Causal Interventions: Conduct targeted interventions for design purposes,
    --Physical Constraints Evaluation: Evaluate constraints to uncover causal mechanisms in material physics.
It is implemented to find atomistic mechanisms behind reducing swithcing barrier of hybrid improper ferroelectrics. 

## Author

Ayana Ghosh  
Email: research.aghosh@gmail.com  

## Related publication


## Project Structure
The project is organized as follows:

  -datasets/           # Folder containing input datasets

  -requirements.txt    # Python package dependencies

  -scripts/            # Folder containing Python scripts with functions and main execution code

## Folder Descriptions

  scripts/: Key files in this folder include:

    --build_models.py: The main script where causal models are created, and refutability tests are performed

    --perform_interv.py: Script to perform causal interventions

    --functions.py: Contains utility functions for performing causal modeling, visualizations, and refutability tests

  outputs/: Store all the output files generated during the execution of the scripts. It includes
  generated plots from causal models, interventions, datasets and results of refutability tests

## Installation & Navigation

*Clone the repository to local machine*: git clone https://github.com/username/causal-modeling-phys.git

*Navigate to the project folder*: cd causal-modeling-phys

*Create a virtual environment to manage the project's dependencies*: python3 -m venv venv

*Activate the virtual environment*:

  On Windows: venv\Scripts\activate; On macOS/Linux: source venv/bin/activate

*Install the required dependencies from requirements.txt*: pip install -r requirements.txt

*Running the Main Script*:
One can execute the main script build_models.py to perform causal modeling, refutability tests and interventions
To run the script, use the following command:

python scripts/build_models.py; python perform_interv.py

## Requirements
The project has the following Python dependencies, which are listed in requirements.txt:

    --numpy (for numerical operations)
    --pandas (for data manipulation)
    --matplotlib (for creating plots)
    --scikit-learn (for machine learning models)
    --graphviz (for visualizing causal models)

