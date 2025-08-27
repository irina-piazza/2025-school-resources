# plot_learning_curve.py

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

def load_and_plot_learning_curve(file_path: Path, output_dir: Path, ignore_initial_epochs: int = 100) -> pd.DataFrame:
    """
    Loads learning curve data, plots it, calculates minimum RMSE after a
    certain number of epochs, and saves the plot as an HTML file.
    """
    # ... (File reading logic remains the same) ...
    with file_path.open("r") as f:
        for line in f:
            if line.lower().startswith("#    epoch"):
                header_line = line
                break
        else:
            raise ValueError(f"Header row not found in {file_path.name}")
            
    header_line = header_line.strip("#").strip()
    column_names = header_line.split()
    data = pd.read_csv(file_path, delim_whitespace=True, comment="#", header=None)
    data.columns = column_names

    print(f"\n- Columns read from {file_path.name}: {data.columns.tolist()}")

    if len(data) > ignore_initial_epochs:
        data_for_min_calc = data.iloc[ignore_initial_epochs:]
    else:
        data_for_min_calc = data

    fig = go.Figure()

    # Plot and analyze Train RMSE
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

    # Plot and analyze Test RMSE
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

    # Update layout with larger font sizes
    fig.update_layout(
        title=dict(
            text=f"Learning Curve - {file_path.stem}",
            font=dict(size=24),  # Title font size
            x=0.5
        ),
        xaxis=dict(
            title_text="Epoch",
            title_font=dict(size=18),  # X-axis title font size
            tickfont=dict(size=14)     # X-axis tick label size
        ),
        yaxis=dict(
            title_text="RMSE (physical units)",
            title_font=dict(size=18),  # Y-axis title font size
            tickfont=dict(size=14),    # Y-axis tick label size
            type="log"
        ),
        legend=dict(
            title_text="Legend",
            title_font=dict(size=18), # Legend title font size
            font=dict(size=16)        # Legend item font size
        ),
        template="plotly_white"
    )

    # Save the plot as an interactive HTML file
    output_file = output_dir / (file_path.stem + ".html")
    fig.write_html(output_file)
    print(f"- Learning curve plot saved to: {output_file}")

    return data