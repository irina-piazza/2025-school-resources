# n2p2_converter.py

# ==============================================================================
#  N2P2 DATA CONVERSION LIBRARY
# ==============================================================================

def _ase_to_n2p2_string(atoms_obj, energy, forces, step, temp):
    """
    Converts a single ASE snapshot to the n2p2 string format.
    """
    lines = []
    lines.append("begin")
    lines.append(f"comment Step: {step}, Temp: {temp:.2f} K, Energy: {energy:.6f} eV")
    cell = atoms_obj.get_cell()
    for i in range(3):
        lines.append(f"lattice {cell[i, 0]:.6f} {cell[i, 1]:.6f} {cell[i, 2]:.6f}")
    
    positions = atoms_obj.get_positions()
    symbols = atoms_obj.get_chemical_symbols()
    for i in range(len(atoms_obj)):
        pos = positions[i]
        sym = symbols[i]
        f = forces[i]
        line = (f"atom {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {sym} 0.0 0.0 "
                f"{f[0]:.6f} {f[1]:.6f} {f[2]:.6f}")
        lines.append(line)
    
    lines.append(f"energy {energy:.6f}")
    lines.append("charge 0.0")
    lines.append("end")
    return "\n".join(lines) + "\n"

def write_n2p2_data(traj_data, output_filename, sampling_rate=1):
    """
    Converts a list of trajectory snapshots to n2p2 format and writes to a file,
    with an optional sampling rate.
    
    Args:
        traj_data (list): List of snapshot dictionaries from the simulation.
        output_filename (str or Path): The name of the file to save the data to.
        sampling_rate (int): The interval at which to sample the trajectory.
                             Defaults to 1 (all frames).
    """
    subsampled_data = traj_data[::sampling_rate]
    
    print(f"\n- Converting {len(subsampled_data)} of {len(traj_data)} snapshots to n2p2 format (sampling rate: {sampling_rate})...")
    print(f"- Writing to file: {output_filename}")

    with open(output_filename, "w") as f:
        for snapshot in subsampled_data:
            n2p2_str = _ase_to_n2p2_string(
                atoms_obj=snapshot["atoms"],
                energy=snapshot["energy"],
                forces=snapshot["forces"],
                step=snapshot["step"],
                temp=snapshot["temperature"]
            )
            f.write(n2p2_str)

    print(f"- Conversion complete. {len(subsampled_data)} structures written.")