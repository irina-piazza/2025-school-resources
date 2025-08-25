# plot_results.py

# ==============================================================================
#  SIMULATION RESULTS - VISUALIZATION
# ==============================================================================

import plotly.graph_objects as go

def plot_potential_energy(traj_data, output_filename="potential_energy_plot.html"):
    """
    Creates an interactive plot of the potential energy from the
    simulation trajectory and saves it to an HTML file.
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

    # 4. Update the layout with larger font sizes
    fig.update_layout(
        title=dict(
            text='<b>Potential Energy Evolution During Simulation</b>',
            font=dict(size=36),  # Title font size
            x=0.5  # Center title
        ),
        xaxis=dict(
            title_text='<b>Simulation Step',
            title_font=dict(size=30),  # X-axis title font size
            tickfont=dict(size=30)     # X-axis tick label size
        ),
        yaxis=dict(
            title_text='<b>Potential Energy (eV)</b>',
            title_font=dict(size=30),  # Y-axis title font size
            tickfont=dict(size=30)     # Y-axis tick label size
        ),
        legend=dict(
            font=dict(size=20) # Legend item font size
        ),
        template='plotly_white'
    )

    # 5. Save the plot to an HTML file
    fig.write_html(output_filename)
    print(f"- Plot successfully saved to: {output_filename}")
    #fig.show()

