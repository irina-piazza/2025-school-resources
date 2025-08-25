# nnp_scaling_manager.py

# ==============================================================================
#  N2P2 SCALING PROCESS MANAGER
# ==============================================================================

import subprocess
import os
from pathlib import Path
import contextlib
import shutil  # <-- Import shutil to search for executables

def find_nnp_executable(base_path):
    """
    Finds the nnp-scaling executable.

    Searches first in a project-local path, then in the system's PATH.

    Args:
        base_path (Path): The project's root directory.

    Returns:
        Path: The absolute path to the nnp-scaling executable.

    Raises:
        FileNotFoundError: If the executable cannot be found.
    """
    # 1. Check for a project-local executable first
    local_path = base_path / "n2p2" / "bin" / "nnp-scaling"
    if local_path.is_file():
        return local_path

    # 2. If not found, search the system's PATH
    system_path_str = shutil.which('nnp-scaling')
    if system_path_str:
        return Path(system_path_str)

    # 3. If still not found, raise an error
    raise FileNotFoundError(
        "Could not find 'nnp-scaling' executable. Please ensure it is "
        "either in your system's PATH or located at './n2p2/bin/nnp-scaling'."
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

def run_scaling(base_path, nnp_executable_path, num_processors=32):
    """
    Sets up the scaling directory, links necessary files, and runs nnp-scaling.
    """
    scaling_dir = base_path / "scaling"
    input_nn_src = base_path / "input_data" / "input.nn"
    input_data_src = base_path / "input_data" / "input.data"

    print("\n- Preparing for nnp-scaling...")
    if not input_data_src.is_file():
        raise FileNotFoundError(f"Training data 'input.data' not found at: {input_data_src}.")
    if not input_nn_src.is_file():
        raise FileNotFoundError(f"Configuration file 'input.nn' not found at: {input_nn_src}.")

    scaling_dir.mkdir(exist_ok=True)
    print(f"- Created scaling directory: {scaling_dir}")

    input_nn_dest = scaling_dir / "input.nn"
    input_data_dest = scaling_dir / "input.data"

    if not input_nn_dest.exists():
        input_nn_dest.symlink_to(input_nn_src)
        print(f"- Linked {input_nn_src.name} to scaling directory.")
    
    if not input_data_dest.exists():
        input_data_dest.symlink_to(input_data_src)
        print(f"- Linked {input_data_src.name} to scaling directory.")

    command = [
        'mpirun', '-np', str(num_processors),
        str(nnp_executable_path), '150'
    ]
    
    print(f"\n- Running command: {' '.join(command)}")
    print("-" * 60)
    
    with working_directory(scaling_dir):
        result = subprocess.run(command, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print("--- STDERR ---")
            print(result.stderr)
            result.check_returncode()
            
    print("-" * 60)