# plot_results.py

# ==============================================================================
#  SIMULATION RESULTS - VISUALIZATION
# ==============================================================================

import plotly.graph_objects as go

def plot_potential_energy(traj_data, output_filename="potential_energy_plot.html"):
    """
    Creates an interactive plot of the potential energy from the
    simulation trajectory and saves it to an HTML file.

    Args:
        traj_data (list): A list of snapshots from the simulation.
        output_filename (str): The name of the output HTML file.
    """
    if not traj_data:
        print("\n- No trajectory data to plot.")
        return

    print(f"\n- Generating the potential energy plot...")

    # 1. Extract data from the trajectory
    steps = [snapshot['step'] for snapshot in traj_data]
    energies = [snapshot['energy'] for snapshot in traj_data]

    # 2. Create a standard figure object
    fig = go.Figure()

    # 3. Add the trace for Potential Energy
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=energies,
            name='Potential Energy',
            mode='lines',
            marker_color='royalblue'
        )
    )

    # 4. Update the layout (titles, labels, etc.)
    fig.update_layout(
        title_text='<b>Potential Energy Evolution During Simulation</b>',
        title_x=0.5,
        xaxis_title='Simulation Step',
        yaxis_title='<b>Potential Energy (eV)</b>',
        template='plotly_white'
    )

    # 5. Save the plot to an HTML file
    fig.write_html(output_filename)
    print(f"- Plot successfully saved to: {output_filename}")