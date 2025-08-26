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
    
    This function prepares a complete, ready-to-use simulation object by:
    1. Ensuring the interatomic potential file is available.
    2. Building a large block of Aluminum atoms (a "supercell").
    3. Configuring the LAMMPS physics engine with the potential.
    4. Attaching the engine (calculator) to the atoms object.

    Args:
        potential_filepath (Path): The path to the interatomic potential file.
        vscale (float): A volume scaling factor to adjust the crystal's density.
    
    Returns:
        ase.Atoms: A fully configured ASE Atoms object ready for simulation.
    """
    # --- Step 1: Ensure the Interatomic Potential File is Available ---
    # The potential file defines the "physics" of the simulation by describing
    # the forces between atoms. For metals like Aluminum, the Embedded Atom
    # Method (EAM) is a highly effective model.
    potential_url = "https://www.ctcms.nist.gov/potentials/Download/2009--Zhakhovskii-V-V-Inogamov-N-A-Petrov-Y-V-et-al--Al/2/Al-2009.eam.alloy"
    
    # This helper function checks if the file exists and is valid. If not, it
    # downloads it from the specified URL. This makes the script portable.
    _download_potential_if_needed(potential_filepath, potential_url)
    
    # This is a final safety check. If the file is still missing after the
    # download attempt, the program stops with a clear error, preventing a more
    # cryptic crash from LAMMPS later on.
    assert potential_filepath.is_file(), f"Potential file is missing: {potential_filepath}!"

    # --- Step 2: Build the Atomic System (Supercell) ---
    # This line uses the Atomic Simulation Environment (ASE) library to construct
    # the virtual block of Aluminum we want to simulate.
    # It's a two-part process:
    # 1. bulk(...): Creates a small, perfect repeating crystal unit cell.
    #    - 'Al': Specifies Aluminum. ASE knows its default crystal structure is FCC.
    #    - a=...: Sets the lattice constant (the side length of the cubic unit cell).
    #      - 4.0495 is the experimental value for Aluminum in Angstroms.
    #      - vscale**(1/3) adjusts this length. Since vscale is a *volume*
    #        scale factor, we take the cube root to apply it to a length.
    # 2. .repeat((5, 5, 5)): Duplicates the small unit cell to build a larger
    #    "supercell" (5x5x5 unit cells). This is crucial for simulating a
    #    bulk material and minimizing boundary effects.
    atoms = bulk('Al', cubic=True, a=4.0495 * vscale**(1 / 3)).repeat((5, 5, 5))

    # --- Step 3: Configure the LAMMPS Calculator (the "Physics Engine") ---
    # This creates an instance of LAMMPSlib, which is ASE's interface to the
    # powerful LAMMPS MD engine.
    lmp = LAMMPSlib(
        # lmpcmds is a list of commands sent directly to LAMMPS for setup.
        lmpcmds=[
            # "pair_style": Tells LAMMPS what kind of physics to use.
            # 'eam/alloy' is the style required for our potential file.
            "pair_style eam/alloy",
            
            # "pair_coeff": Links the physics style to the actual potential file.
            # It tells LAMMPS: "For all pairs of atoms (* *), use the parameters
            # found in the file at `potential_filepath` for the element 'Al'".
            f"pair_coeff * * {str(potential_filepath)} Al" 
        ],
        
        # LAMMPS uses numerical atom types (1, 2, ...), not chemical symbols.
        # This line maps ASE's 'Al' symbol to LAMMPS's atom type 1.
        atom_types={'Al': 1},
        
        # Suppress the default LAMMPS log file (log.lammps).
        log_file=None,
        
        # Performance optimization: keeps the LAMMPS process running in the
        # background, avoiding costly restarts at every simulation step.
        keep_alive=True
    )

    # --- Step 4: Attach the Calculator to the Atoms and Return ---
    # This line connects the atoms object to the physics engine. Now, whenever a
    # physical property (like forces or energy) is requested from `atoms`,
    # it will automatically use the `lmp` calculator to compute it.
    atoms.calc = lmp
    
    # Return the fully configured simulation object.
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