# Step 1: Install the necessary libraries
# Already installed in a previous cell

# Step 2: Import the libraries
import ovito.io
from ovito.modifiers import CoordinationAnalysisModifier # Correct import
import plotly.graph_objects as go
import numpy as np # Make sure numpy is imported
from pathlib import Path
def generate_rdf_animation(file_path: str, 
                           output_filename: str = 'rdf_animation.html', 
                           cutoff_radius: float = 8.0, 
                           number_of_bins: int = 150,
                           frame_duration: int = 100):
    """
    Calculates and visualizes an animation of the Radial Distribution Function (RDF)
    from a trajectory file and saves it as an interactive HTML file.

    Args:
        file_path (str): The full path to the trajectory file (e.g., .lammpstrj).
        output_filename (str, optional): The name for the output HTML file. 
                                         Defaults to 'rdf_animation.html'.
        cutoff_radius (float, optional): The maximum cutoff radius in Angstroms for the 
                                         RDF calculation. Defaults to 8.0.
        number_of_bins (int, optional): The number of bins for the RDF histogram. 
                                        Defaults to 150.
        frame_duration (int, optional): The duration of each frame in the animation in 
                                        milliseconds. Defaults to 100.

    Returns:
        go.Figure: The Plotly Figure object containing the animation. 
                   Returns None if no valid RDF data is found.
    """
    
    # --- Load data and set up the OVITO pipeline ---
    print(f"Loading file: {file_path}")
    try:
        pipeline = ovito.io.import_file(file_path, multiple_frames=True)
        num_frames = pipeline.source.num_frames
        if num_frames == 0:
            print("Error: OVITO found no frames in the file.")
            return None
        print(f"Found {num_frames} frames in the trajectory.")
    except Exception as e:
        print(f"Error while loading file with OVITO: {e}")
        return None

    # Add the modifier to calculate the RDF
    rdf_modifier = CoordinationAnalysisModifier(
        cutoff=cutoff_radius,
        number_of_bins=number_of_bins
    )
    pipeline.modifiers.append(rdf_modifier)

    # --- Pre-calculate the RDF for all frames ---
    all_rdf_data = []
    print("Calculating RDF for all frames...")
    for frame_index in range(num_frames):
        data = pipeline.compute(frame_index)
        if 'coordination-rdf' in data.tables and data.tables['coordination-rdf'].y is not None:
            rdf_table = data.tables['coordination-rdf']
            frame_data = {'x': rdf_table.xy()[:, 0], 'y': rdf_table.xy()[:, 1], 'original_frame': frame_index}
            all_rdf_data.append(frame_data)
        else:
            # Add a placeholder to maintain index correspondence if needed
            all_rdf_data.append(None) 

        if (frame_index + 1) % 50 == 0 or frame_index == num_frames - 1:
            print(f"  Processed frame {frame_index + 1}/{num_frames}")

    # --- Filter data and create the plot ---
    valid_rdf_data = [rdf for rdf in all_rdf_data if rdf is not None and len(rdf['x']) > 0]

    if not valid_rdf_data:
        print("Error: No valid RDF data was collected from the trajectory.")
        return None
    
    print("RDF calculation complete. Building interactive plot...")

    # Find the max y-value to set the axis range dynamically
    max_y_value = max(np.max(rdf['y']) for rdf in valid_rdf_data)
    
    # Create the animation frames
    frames = [go.Frame(
        data=[go.Scatter(x=rdf['x'], y=rdf['y'], mode='lines', name='g(r)')],
        name=str(rdf['original_frame']),
        layout=go.Layout(title_text=f"Animated RDF - Original Frame {rdf['original_frame']}")
    ) for rdf in valid_rdf_data]

    # Create the slider steps
    slider_steps = [dict(
        method="animate",
        args=[[str(rdf['original_frame'])],
              {"frame": {"duration": frame_duration, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
        label=str(rdf['original_frame'])
    ) for rdf in valid_rdf_data]

    # Define the plot layout
    initial_data = valid_rdf_data[0]
    layout = go.Layout(
        xaxis=dict(range=[0, cutoff_radius], title="Distance (r) [Å]"),
        yaxis=dict(range=[0, max_y_value * 1.1], title="g(r)"),
        title=f"Animated RDF - Original Frame {initial_data['original_frame']}",
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(label="Play", method="animate", args=[None, {"frame": {"duration": frame_duration, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]),
                dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])
            ]
        )],
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "Frame: "},
            pad={"t": 50},
            steps=slider_steps
        )]
    )

    # Assemble the final figure
    fig = go.Figure(
        data=[go.Scatter(x=initial_data['x'], y=initial_data['y'], mode='lines', name='g(r)')],
        layout=layout,
        frames=frames
    )
    
    # --- Save and return the figure ---
    fig.write_html(output_filename)
    print(f"Animation successfully saved as '{output_filename}'")
    return fig

# --- EXAMPLE USAGE IN GOOGLE COLAB ---
if __name__ == '__main__':
    # Define the path to your file here
    colab_file_path = 'dump_equil.lammpstrj'
    
    # Call the function with the desired parameters
    my_rdf_figure = generate_rdf_animation(
        file_path=colab_file_path,
        output_filename='rdf_animation_Al.html',
        cutoff_radius=8.0,
        number_of_bins=200
    )

    # If you wish, you can display the figure directly in the notebook after creating it
    #if my_rdf_figure:
    #    my_rdf_figure.show()
        