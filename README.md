# Bayesian Optimization for Materials Science Reproduction

This repository contains a one-file Python script to reproduce the results comparing Gaussian Process with Automatic Relevance Determination (GP-ARD) against Random Selection for materials discovery, as outlined in the provided context.

## Setup

A `requirements.txt` file is included with all necessary Python packages. You can use the provided setup scripts to create a virtual environment and install these dependencies.

### Windows Setup

**Using Command Prompt:**

```cmd
setup.cmd
```

**Using PowerShell:**

```powershell
.\setup.ps1
```

### Manual Setup (All Platforms)

1.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    ```

2.  **Activate the environment:**

    *   **Windows (cmd):** `venv\Scripts\activate.bat`
    *   **Windows (PowerShell):** `.\venv\Scripts\Activate.ps1`
    *   **Linux/macOS:** `source venv/bin/activate`

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Running the Analysis

Once the setup is complete, you can run the main script:

```bash
python reproduce_analysis.py
```

The script will perform the following steps:

1.  Load and process the "Crossed barrel" and "AgNP" material science datasets.
2.  For each dataset, run two benchmark loops over 50 random seeds:
    *   **GP-ARD:** An active learning loop using a Gaussian Process with a Matérn 5/2 ARD kernel and a Lower Confidence Bound acquisition function.
    *   **Random Selection:** A baseline loop that randomly selects new candidates.
3.  Generate and save two plots for each dataset:
    *   `[Dataset]_discovery_rate.png`: A plot showing the Top 5% Coverage vs. the number of experiments, comparing the median performance and Interquartile Range (IQR) of GP-ARD and Random Selection.
    *   `[Dataset]_manifold.png`: A PCA projection of the dataset's feature space, colored by the target objective value, to visualize the distribution of high-performing materials.

The resulting plots will be saved in the root directory.
