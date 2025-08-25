# md_simulation_library.py

import urllib.request
import numpy as np
from pathlib import Path
import shutil # <-- Import shutil for efficient file copying

# Importing ASE functionalities
from ase import units
from ase.build import bulk
from ase.calculators.lammpslib import LAMMPSlib
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.nvtberendsen import NVTBerendsen


def _download_potential_if_needed(filepath, url):
    """
    Checks if a file exists and downloads it, adding a User-Agent header
    to prevent HTTP 403 Forbidden errors.
    """
    if not filepath.is_file():
        print(f"- Potential file '{filepath.name}' not found. Downloading...")
        try:
            # Create a request object with a common browser User-Agent header
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            request = urllib.request.Request(url, headers=headers)
            
            # Open the URL and save the content to the file
            with urllib.request.urlopen(request) as response, open(filepath, 'wb') as out_file:
                # Use shutil.copyfileobj to efficiently stream the download to the file
                shutil.copyfileobj(response, out_file)
                
            print(f"- Download complete.")
        except urllib.error.HTTPError as e:
            print(f"Error: Could not download the potential file. The server returned an error.")
            print(f"Details: {e}")
            raise FileNotFoundError(f"Failed to download required potential file: {filepath}")
        except Exception as e:
            print(f"An unexpected error occurred during download: {e}")
            raise
    else:
        print(f"- Potential file '{filepath.name}' found.")


def setup_simulation(potential_filepath, vscale):
    """
    Sets up the atomic supercell and the LAMMPS calculator.
    Accepts a pathlib.Path object for the potential file.
    """
    potential_url = "https://www.ctcms.nist.gov/potentials/Download/2009--Zhakhovskii-V-V-Inogamov-N-A-Petrov-Y-V-et-al--Al/2/Al-2009.eam.alloy"
    
    _download_potential_if_needed(potential_filepath, potential_url)
    
    assert potential_filepath.is_file(), f"Potential file is missing: {potential_filepath}!"

    atoms = bulk('Al', cubic=True, a=4.0495 * vscale**(1 / 3)).repeat((5, 5, 5))

    lmp = LAMMPSlib(
        lmpcmds=[
            "pair_style eam/alloy",
            f"pair_coeff * * {str(potential_filepath)} Al" 
        ],
        atom_types={'Al': 1},
        log_file=None,
        keep_alive=True
    )
    atoms.calc = lmp
    return atoms

def initialize_dynamics(atoms, T_start, timestep_fs):
    """
    Initializes velocities and creates the dynamics object.
    """
    MaxwellBoltzmannDistribution(atoms, temperature_K=T_start)
    ZeroRotation(atoms)
    Stationary(atoms)
    dyn = NVTBerendsen(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=T_start,
        taut=10 * units.fs
    )
    return dyn

def run_simulation_cycle(dyn, atoms, n_steps, temperatures):
    """
    Runs the molecular dynamics simulation cycle and collects the data.
    """
    traj_data = []
    def log_step():
        step_idx = dyn.nsteps
        if step_idx >= n_steps: return
        T_current = temperatures[step_idx]
        dyn.set_temperature(T_current)
        traj_data.append({
            "step": step_idx,
            "atoms": atoms.copy(),
            "energy": atoms.get_potential_energy(),
            "temperature": atoms.get_temperature(),
            "forces": atoms.get_forces()
        })
    dyn.attach(log_step, interval=1)
    print("- Running MD...")
    dyn.run(n_steps)
    return traj_data