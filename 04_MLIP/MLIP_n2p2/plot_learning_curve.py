# plot_learning_curve.py

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

def load_and_plot_learning_curve(file_path: Path, output_dir: Path, ignore_initial_epochs: int = 100, y_min=0.007, y_max=0.013) -> pd.DataFrame:
    """
    Loads learning curve data, plots it, calculates minimum RMSE after a
    certain number of epochs, and saves the plot as an HTML file.

    Args:
        file_path: Path to the 'learning-curve*.out' file.
        output_dir: Directory to save the plot.
        ignore_initial_epochs: Number of initial epochs to ignore when finding the minimum RMSE.
        y_min: Minimum y-axis limit (currently not used in log scale).
        y_max: Maximum y-axis limit (currently not used in log scale).

    Returns:
        A pandas DataFrame containing the loaded data.
    """
    # Read the file to find the header row
    with file_path.open("r") as f:
        for line in f:
            if line.lower().startswith("#    epoch"):
                header_line = line
                break
        else:
            raise ValueError(f"Header row not found in {file_path.name}")

    # Clean up the header
    header_line = header_line.strip("#").strip()
    column_names = header_line.split()

    # Read the data, ignoring comment lines
    data = pd.read_csv(file_path, delim_whitespace=True, comment="#", header=None)

    if len(column_names) != data.shape[1]:
        raise ValueError(
            f"Mismatch between number of columns ({len(column_names)}) and read columns ({data.shape[1]})"
        )
    data.columns = column_names

    print(f"\n- Columns read from {file_path.name}: {data.columns.tolist()}")

    # Create a sliced DataFrame for calculating the minimum, ignoring initial epochs
    if len(data) > ignore_initial_epochs:
        data_for_min_calc = data.iloc[ignore_initial_epochs:]
    else:
        data_for_min_calc = data # Not enough data to ignore, use all of it

    # Create a Plotly figure
    fig = go.Figure()

    # Plot RMSEpa_Etrain_pu and find its minimum
    if "RMSEpa_Etrain_pu" in data.columns:
        fig.add_trace(go.Scatter(
            x=data["epoch"],
            y=data["RMSEpa_Etrain_pu"],
            mode='lines',
            name="Train RMSE/atom"
        ))
        if not data_for_min_calc.empty:
            min_val = data_for_min_calc["RMSEpa_Etrain_pu"].min()
            min_epoch = data_for_min_calc.loc[data_for_min_calc["RMSEpa_Etrain_pu"].idxmin(), "epoch"]
            print(f"  - Min Train RMSE/atom (after epoch {ignore_initial_epochs}): {min_val:.6f} at epoch {min_epoch}")

    # Plot RMSEpa_Etest_pu and find its minimum
    if "RMSEpa_Etest_pu" in data.columns:
        fig.add_trace(go.Scatter(
            x=data["epoch"],
            y=data["RMSEpa_Etest_pu"],
            mode='lines',
            name="Test RMSE/atom"
        ))
        if not data_for_min_calc.empty:
            min_val = data_for_min_calc["RMSEpa_Etest_pu"].min()
            min_epoch = data_for_min_calc.loc[data_for_min_calc["RMSEpa_Etest_pu"].idxmin(), "epoch"]
            print(f"  - Min Test RMSE/atom (after epoch {ignore_initial_epochs}): {min_val:.6f} at epoch {min_epoch}")

    # Update layout for title, labels, and scales
    fig.update_layout(
        title=f"Learning Curve - {file_path.stem}",
        xaxis_title="Epoch",
        yaxis_title="RMSE (physical units)",
        yaxis_type="log",
        legend_title="Legend",
        template="plotly_white"
    )

    # Save the plot as an interactive HTML file
    output_file = output_dir / (file_path.stem + ".html")
    fig.write_html(output_file)
    print(f"- Learning curve plot saved to: {output_file}")

    return data

# This block allows the script to be run directly for testing
if __name__ == '__main__':
    print("Running plot_learning_curve.py in standalone mode for testing...")
    base_dir = Path(__file__).parent
    training_dir = base_dir / "training"
    output_dir = base_dir / "output_plot"
    training_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    test_file_path = training_dir / "learning-curve.out"

    if not test_file_path.exists():
        print(f"Error: The test file was not found at '{test_file_path}'")
    else:
        print(f"Loading and plotting file: {test_file_path.name}")
        # Test the new parameter
        load_and_plot_learning_curve(test_file_path, output_dir, ignore_initial_epochs=100)
        print("-" * 40)
        print(f"Test complete. Check for '{test_file_path.stem}.html' in the '{output_dir}' folder.")
