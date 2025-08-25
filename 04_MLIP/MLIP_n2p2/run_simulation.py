# run_simulation.py

# ==============================================================================
#  MAIN SCRIPT TO RUN MD SIMULATION AND POST-PROCESSING
# ==============================================================================

import time
import numpy as np
from pathlib import Path
import subprocess

# Import from our custom libraries
from md_simulation_library import setup_simulation, initialize_dynamics, run_simulation_cycle
from n2p2_converter import write_n2p2_data
from plot_results import plot_potential_energy
from nnp_scaling_manager import run_scaling, find_nnp_executable
from nnp_training_manager import run_training
from trajectory_processor import save_trajectory_to_xyz
from plot_learning_curve import load_and_plot_learning_curve

# ==============================================================================
# SECTION: DIRECTORY AND FILE PATHS SETUP
# ==============================================================================

def get_simulation_parameters():
    """
    Defines and returns all simulation parameters using absolute paths.
    """
    base_path = Path.cwd()
    
    input_dir = base_path / "input_data"
    plot_dir = base_path / "output_plot"
    scaling_dir = base_path / "scaling"
    training_dir = base_path / "training"

    params = {
        # File paths
        "potential_filepath": input_dir / "Al-2009.eam.alloy",
        "n2p2_output_filepath": input_dir / "input.data",
        "plot_output_filepath": plot_dir / "potential_energy_plot.html",
        # Directories to create
        "directories": [input_dir, plot_dir, scaling_dir, training_dir],
        # Simulation parameters
        "T_start": 100,
        #"T_end": 1000,
        #"n_steps": 2000,
        "T_end": 2000,
        "n_steps": 20000,
        "timestep_fs": 1.5,
        "vscale": (2.697 / 2.304),
        # NNP process parameters
        "scaling_processors": 16,
        "training_processors": 8,
        # Workflow control
        "skip_nnp_if_exists": True,
        "ignore_epochs_for_min": 100, # <-- PARAMETER for ignoring initial epochs
    }
    return params

# ... (create_directories, run_workflow, and report_results functions remain unchanged) ...
def create_directories(dir_paths):
    print("- Checking and creating project directories...")
    for path in dir_paths:
        path.mkdir(parents=True, exist_ok=True)
    print("- Directory setup complete.")

def run_workflow(params):
    atoms = setup_simulation(params["potential_filepath"], params["vscale"])
    print(f"- Setting up NVT temperature ramp from {params['T_start']} K to {params['T_end']} K over {params['n_steps']} steps...")
    dyn = initialize_dynamics(atoms, params["T_start"], params["timestep_fs"])
    temperatures = np.linspace(params["T_start"], params["T_end"], params["n_steps"])
    traj_data = run_simulation_cycle(dyn, atoms, params["n_steps"], temperatures)
    return traj_data

def report_results(start_time, traj_data):
    end_time = time.time()
    print(f"- MD finished. Total time: {end_time - start_time:.2f} s")
    print(f"- Collected {len(traj_data)} snapshots.")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main function that orchestrates the entire workflow."""
    t0 = time.time()
    
    simulation_params = get_simulation_parameters()
    create_directories(simulation_params["directories"])
    
    trajectory_data = run_workflow(simulation_params)
    report_results(t0, trajectory_data)

    if trajectory_data:
        xyz_output_dir = simulation_params["n2p2_output_filepath"].parent
        save_trajectory_to_xyz(
            trajectory_data,
            xyz_output_dir,
            simulation_params["T_start"],
            simulation_params["T_end"]
        )
        
        write_n2p2_data(trajectory_data, simulation_params["n2p2_output_filepath"], sampling_rate=100)

        try:
            base_path = Path.cwd()
            scaling_dir = base_path / "scaling"
            training_dir = base_path / "training"
            
            # --- Conditionally run nnp-scaling ---
            if not (simulation_params["skip_nnp_if_exists"] and scaling_dir.is_dir() and any(scaling_dir.iterdir())):
                nnp_scaling_executable = find_nnp_executable(base_path)
                print(f"- Found nnp-scaling executable at: {nnp_scaling_executable}")
                run_scaling(
                    base_path,
                    nnp_scaling_executable,
                    simulation_params["scaling_processors"]
                )
                print("- nnp-scaling completed successfully.")
            else:
                print("\n- 'scaling' directory already exists. Skipping nnp-scaling process.")

            # --- Conditionally run nnp-training ---
            if not (simulation_params["skip_nnp_if_exists"] and training_dir.is_dir() and any(training_dir.iterdir())):
                run_training(
                    base_path,
                    simulation_params["training_processors"]
                )
                print("- nnp-training completed successfully.")
            else:
                print("- 'training' directory already exists. Skipping nnp-training process.")
            
            # --- Plot learning curve now that training is done ---
            learning_curve_file = training_dir / "learning-curve.out"
            if learning_curve_file.is_file():
                load_and_plot_learning_curve(
                    learning_curve_file,
                    simulation_params["plot_output_filepath"].parent,
                    ignore_initial_epochs=simulation_params["ignore_epochs_for_min"] # <-- PASS THE PARAMETER
                )
            else:
                print(f"- Warning: Learning curve file not found at '{learning_curve_file}'. Cannot plot.")

        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            print(f"\n- An error occurred during the n2p2 process: {e}")
        
        # --- Create the potential energy plot ---
        plot_potential_energy(
            trajectory_data,
            simulation_params["plot_output_filepath"]
        )
            
    else:
        print("\n- No trajectory data to process.")

if __name__ == "__main__":
    main()