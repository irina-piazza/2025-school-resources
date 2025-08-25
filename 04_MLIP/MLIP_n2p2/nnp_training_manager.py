# nnp_training_manager.py

# ==============================================================================
#  N2P2 TRAINING PROCESS MANAGER
# ==============================================================================

import subprocess
import os
from pathlib import Path
import contextlib
import shutil

def find_nnp_train_executable(base_path):
    """
    Finds the nnp-train executable.

    Searches first in a project-local path, then in the system's PATH.

    Args:
        base_path (Path): The project's root directory.

    Returns:
        Path: The absolute path to the nnp-train executable.

    Raises:
        FileNotFoundError: If the executable cannot be found.
    """
    # 1. Check for a project-local executable first
    local_path = base_path / "n2p2" / "bin" / "nnp-train"
    if local_path.is_file():
        return local_path

    # 2. If not found, search the system's PATH
    system_path_str = shutil.which('nnp-train')
    if system_path_str:
        return Path(system_path_str)

    # 3. If still not found, raise an error
    raise FileNotFoundError(
        "Could not find 'nnp-train' executable. Please ensure it is "
        "either in your system's PATH or located at './n2p2/bin/nnp-train'."
    )


@contextlib.contextmanager
def working_directory(path):
    """Changes the working directory and returns to the original when done."""
    original_dir = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(original_dir)


def run_training(base_path, num_processors=16):
    """
    Sets up the training directory, links necessary files, and runs nnp-train.
    
    Args:
        base_path (Path): The absolute path to the project's root directory.
        num_processors (int): Number of MPI processes to use for training.
    """
    # 1. Find the executable first
    nnp_train_executable = find_nnp_train_executable(base_path)
    print(f"- Found nnp-train executable at: {nnp_train_executable}")
    
    # 2. Define all necessary paths
    training_dir = base_path / "training"
    # Canonical sources for the links
    input_nn_src = base_path / "input_data" / "input.nn"
    input_data_src = base_path / "input_data" / "input.data"
    scaling_data_src = base_path / "scaling" / "scaling.data"

    # 3. Check for required files before starting
    print("\n- Preparing for nnp-training...")
    for f in [input_nn_src, input_data_src, scaling_data_src]:
        if not f.is_file():
            raise FileNotFoundError(f"Required file for training not found: {f}")

    # 4. Create the training directory
    training_dir.mkdir(exist_ok=True)
    print(f"- Created training directory: {training_dir}")
    
    # 5. Create symbolic links inside the training directory
    links = {
        "input.nn": input_nn_src,
        "input.data": input_data_src,
        "scaling.data": scaling_data_src
    }
    
    for dest_name, src_path in links.items():
        dest_path = training_dir / dest_name
        if not dest_path.exists():
            dest_path.symlink_to(src_path)
            print(f"- Linked {src_path.name} to training directory.")
            
    # 6. Run the training command
    command = [
        'mpirun', '-np', str(num_processors),
        str(nnp_train_executable)
    ]
    
    print(f"\n- Running command: {' '.join(command)}")
    print("-" * 60)
    
    with working_directory(training_dir):
        result = subprocess.run(command, capture_output=True, text=True)
        
        print(result.stdout)
        
        if result.returncode != 0:
            print("--- STDERR ---")
            print(result.stderr)
            result.check_returncode()
            
    print("-" * 60)