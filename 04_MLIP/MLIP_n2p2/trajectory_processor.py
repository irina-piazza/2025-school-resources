# trajectory_processor.py

# ==============================================================================
#  TRAJECTORY DATA PROCESSING UTILITIES
# ==============================================================================

from ase.io import write
from pathlib import Path

def save_trajectory_to_xyz(traj_data, output_dir, T_start, T_end, stride=100):
    """
    Subsamples a trajectory and saves the atomic frames to an XYZ file.

    Args:
        traj_data (list): The full list of snapshots from the simulation.
        output_dir (Path): The directory where the XYZ file will be saved.
        T_start (int): The starting temperature, used for the filename.
        T_end (int): The ending temperature, used for the filename.
        stride (int): The interval for subsampling the trajectory.
                      Defaults to 100.
    """
    if not traj_data:
        print("\n- No trajectory data to save.")
        return

    # 1. Define the output filename and construct the full path
    xyz_filename = f"traj_ramp_{T_start}K_to_{T_end}K.xyz"
    output_path = output_dir / xyz_filename

    print(f"\n- Subsampling trajectory with a stride of {stride}...")

    # 2. Subsample the trajectory data by extracting the "atoms" object
    #    from every Nth snapshot (where N = stride).
    subsampled_frames = [
        traj_data[i]["atoms"] for i in range(0, len(traj_data), stride)
    ]

    # 3. Write the collected frames to the XYZ file
    write(output_path, subsampled_frames)

    print(f"- Subsampled trajectory with {len(subsampled_frames)} frames saved to: {output_path}")


# def save_xyz_format(T_start, T_end, traj_data):
#     xyz_filename = f"traj_ramp_{T_start}K_to_{T_end}K.xyz"
#     traj_cut=[traj_data[i]["atoms"] for i in list(range(0,len(traj_data)-1,100))] # Printing at each 100 steps
#     write(xyz_filename, [frame for frame in traj_cut])
#     print(f"- Trajectory saved to {xyz_filename}")