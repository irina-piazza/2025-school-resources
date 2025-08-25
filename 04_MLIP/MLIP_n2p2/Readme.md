
# Automated MD Simulation and N2P2 Training Workflow

This project provides a complete, end-to-end Python workflow for running a molecular dynamics (MD) simulation, processing the output, and training a neural network potential (NNP) using the n2p2 package. The entire process is automated through a single command.

## Features

-   **Automated MD Simulation**: Runs a temperature ramp MD simulation for Aluminum using ASE and LAMMPS.
-   **Robust Setup**: Automatically creates the necessary project directory structure. It also downloads the required EAM potential file, checks for corruption, and includes workarounds for common download errors (HTTP 403 Forbidden).
-   **Data Conversion**: Converts the simulation trajectory into the `input.data` format required by n2p2.
-   **Integrated N2P2 Workflow**:
    1.  Automatically runs `nnp-scaling` to generate scaling data.
    2.  Automatically runs `nnp-train` to train the neural network potential.
-   **Dynamic Executable Finding**: Intelligently locates the `nnp-scaling` and `nnp-train` executables, whether they are installed system-wide or within the project directory.
-   **Visualization**: Generates an interactive HTML plot of the potential energy from the simulation using Plotly.
-   **Modular Code**: The project is organized into clean, reusable Python modules for simulation, data conversion, and n2p2 process management.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

1.  **Python** (version 3.9 or higher).
2.  The **`venv` module**, which is typically included with Python.
3.  **An MPI implementation** such as OpenMPI. This is required to run `mpirun`. You can typically install it via your system's package manager (e.g., `sudo apt-get install openmpi-bin` on Debian/Ubuntu).
4.  **A compiled version of n2p2**. The script needs access to the `nnp-scaling` and `nnp-train` executables.

## Installation and Setup

Follow these steps to set up the project environment.

**1. Clone the Repository**
Clone this project to your local machine.
```bash
git clone https://github.com/irina-piazza/2025-school-resources.git
cd 2025-school-resources/04_MLIP/mlip_project
```

**2. Place n2p2 Binaries (if not in PATH)**
If your `n2p2` executables are not in your system's PATH, create a directory structure within the project and place them there. The script will look here first:
```text
<project_root>/
└── n2p2/
    └── bin/
        ├── nnp-scaling
        └── nnp-train
```
In any case you can place `n2p2` whereever you prefer since there will be a function that look for nnp-scaling and nnp-train executable. 
**3. Create and Activate the Python Virtual Environment**
It is highly recommended to use a virtual environment to manage dependencies.
```bash
# Create the virtual environment
python3 -m venv .venv

# Activate the environment
# On Linux or macOS:
source .venv/bin/activate
# On Windows:
# .\.venv\Scripts\activate
```

**4. Install Dependencies**
Install all the required Python packages from the `requirements.txt` file.
```bash
pip install -r requirements.txt
```
or do

```bash
pip install ase plotly pandas lammps==2024.8.29.3.0
```

**5. Create the `input.nn` Configuration File if is not fournished in input_data**
This workflow requires you to provide an `input.nn` file for the n2p2 training. Create a file named `input.nn` inside the `input_data` directory.

The script will automatically create the `input_data` folder, but you must add the file yourself.

**File: `input_data/input.nn`**
```ini
# n2p2 input file - Example for Aluminum (Al)
number_of_elements 1
elements Al

# Neural Network (NN)
global_hidden_layers 3
global_nodes         20 20 20
global_activation  2

# Symmetry Functions
symmetry_function_type BehlerParrinello
atom_energy Al 0.0
r_cut 8.0

# Radial functions (Type 2)
symfunction_short Al 2 0.03 0.0 1.0
symfunction_short Al 2 0.03 0.0 2.0
symfunction_short Al 2 0.03 0.0 3.0
symfunction_short Al 2 0.03 0.0 4.0
symfunction_short Al 2 0.03 0.0 5.0
symfunction_short Al 2 0.03 0.0 6.0
symfunction_short Al 2 0.03 0.0 7.0

# Angular functions (Type 5)
symfunction_short Al 5 0.003 1.0 1.0
symfunction_short Al 5 0.003 1.0 2.0
symfunction_short Al 5 0.003 1.0 4.0
symfunction_short Al 5 0.003 1.0 8.0
symfunction_short Al 5 0.003 -1.0 1.0
symfunction_short Al 5 0.003 -1.0 2.0
symfunction_short Al 5 0.003 -1.0 4.0
symfunction_short Al 5 0.003 -1.0 8.0
```

## Project Structure

The project is organized into several Python modules for clarity and maintainability.

```text
.
├── input_data/
│   └── input.nn                # User-provided n2p2 configuration
├── n2p2/
│   └── bin/
│       ├── nnp-scaling         # (Optional) Local n2p2 executables
│       └── nnp-train
├── md_simulation_library.py    # Core functions for the MD simulation
├── n2p2_converter.py           # Handles conversion of trajectory to n2p2 format
├── nnp_scaling_manager.py      # Manages the nnp-scaling process
├── nnp_training_manager.py     # Manages the nnp-train process
├── plot_results.py             # Generates plots from simulation data
├── requirements.txt            # List of Python dependencies
├── Readme.md                   # Documentation
└── run_simulation.py           # Main script to orchestrate the entire workflow

```

## Usage: Running the Full Workflow

With the environment activated and the `input.nn` file in place, you can run the entire workflow with a single command from the project's root directory:

```bash
python run_simulation.py
```

### Workflow Explained

This command will execute the following steps in sequence:
1.  **Directory Setup**: Creates `input_data`,  `output_plot`, `scaling`, and `training` directories if they don't exist.
2.  **Potential Download**: Downloads the `Al-2009.eam.alloy` potential file into `input_data/`.
3.  **MD Simulation**: Runs the LAMMPS simulation.
4.  **Data Conversion**: Writes the trajectory data to `nnp_train_data/input.data`.
5.  **NNP Scaling**: Sets up the `scaling/` directory with symbolic links and runs `nnp-scaling`.
6.  **NNP Training**: Sets up the `training/` directory with symbolic links and runs `nnp-train`.
7.  **Plot Generation**: Creates `output_plot/potential_energy_plot.html`.

## Output

After a successful run, your project directory will be populated with the following outputs:

-   `input_data/Al-2009.eam.alloy`: The downloaded EAM potential.
-   `input_data/input.data`: The trajectory data ready for training.
-   `scaling/`: Contains symbolic links and output files from `nnp-scaling`, including `scaling.data`.
-   `training/`: Contains symbolic links and all output files from `nnp-train`, such as `weights.*.data` and `train-errors.dat`.
-   `output_plot/potential_energy_plot.html`: An interactive plot of the simulation's potential energy.

## Customization

To modify the simulation or workflow parameters, edit the `get_simulation_parameters` function in the `run_simulation.py` script. You can easily change:
-   Simulation steps (`n_steps`)
-   Temperature range (`T_start`, `T_end`)
-   Sampling_rate, how many snapshots are save from the 20000 original
-   Number of processors for scaling and training (`scaling_processors`, `training_processors`)
-   File and directory names