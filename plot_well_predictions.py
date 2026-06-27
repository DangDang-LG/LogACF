import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
from datetime import datetime
import json

def setup_plotting_environment():
    """
    Configure the Matplotlib plotting environment.

    This setup includes:
    1. Setting the default font family to Arial;
    2. Ensuring text in PDF/SVG outputs remains editable (not converted to paths);
    3. Preventing incorrect rendering of minus signs in plots.

    Returns:
        None
    """
    plt.rcParams.update({
        "font.family": "Arial",

        # Use TrueType fonts in PDF to keep text editable
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # Keep text as text in SVG instead of converting to paths
        "svg.fonttype": "none",

        # Fix minus sign rendering issues
        "axes.unicode_minus": False,
    })

def convert_numpy_types(obj):
    """
    Recursively convert NumPy, Pandas, and other non-serializable types
    into native Python data types for JSON serialization or logging.

    Supported conversions:
    - np.integer → int
    - np.floating → float
    - np.ndarray → list
    - pd.Series → list
    - pd.DataFrame → list of dictionaries
    - dict → recursive conversion
    - list → recursive conversion
    - tuple → recursive conversion
    - NaN / NaT → None

    Parameters:
        obj: Any Python object potentially containing NumPy/Pandas types.

    Returns:
        A fully JSON-serializable Python-native object.
    """

    if isinstance(obj, np.integer):
        return int(obj)

    elif isinstance(obj, np.floating):
        return float(obj)

    elif isinstance(obj, np.ndarray):
        return obj.tolist()

    elif isinstance(obj, pd.Series):
        return obj.tolist()

    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict("records")

    elif isinstance(obj, dict):
        return {
            key: convert_numpy_types(value)
            for key, value in obj.items()
        }

    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]

    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)

    elif pd.isna(obj):
        return None

    else:
        return obj


def load_and_merge_data(model_predictions_path, real_test_result_path):
    """
    Load and merge ground truth data with model prediction results.

    The function reads two CSV files:
    - Real test data (ground truth)
    - Model prediction results

    and merges them using WELLNUM and DEPTH as keys.

    Parameters:
        model_predictions_path (str): Path to the model prediction CSV file.
        real_test_result_path (str): Path to the ground truth CSV file.

    Returns:
        pd.DataFrame: Merged DataFrame containing both real and predicted values.
    """

    model_predictions = pd.read_csv(model_predictions_path)
    real_test_result = pd.read_csv(real_test_result_path)

    merged_df = pd.merge(
        real_test_result,
        model_predictions,
        on=["WELLNUM", "DEPTH"],
        suffixes=("", "_pred")
    )

    return merged_df


def classify_reservoir_quality(phif, sw, vsh, quality_criteria=None):
    """
    Classify reservoir quality based on petrophysical properties.

    The classification is performed using empirical threshold rules
    on porosity (PHIF), water saturation (SW), and shale volume (VSH).

    Parameters:
        phif (float): Effective porosity.
        sw (float): Water saturation.
        vsh (float): Shale volume.
        quality_criteria (dict, optional): Custom classification thresholds.

    Returns:
        str or None: Reservoir quality label:
            - "Excellent"
            - "Good"
            - "Fair"
            - None (non-reservoir or poor quality)
    """

    # Default empirical classification criteria
    if quality_criteria is None:
        quality_criteria = {
            "Excellent": {"phif_min": 0.20, "sw_max": 0.30, "vsh_max": 0.20},
            "Good": {"phif_min": 0.15, "sw_max": 0.40, "vsh_max": 0.30},
            "Fair": {"phif_min": 0.10, "sw_max": 0.50, "vsh_max": 0.35}
        }

    # =========================================================
    # Classification logic (hierarchical rule-based system)
    # =========================================================
    if (
        phif >= quality_criteria["Excellent"]["phif_min"]
        and sw <= quality_criteria["Excellent"]["sw_max"]
        and vsh <= quality_criteria["Excellent"]["vsh_max"]
    ):
        return "Excellent"

    elif (
        phif >= quality_criteria["Good"]["phif_min"]
        and sw <= quality_criteria["Good"]["sw_max"]
        and vsh <= quality_criteria["Good"]["vsh_max"]
    ):
        return "Good"

    elif (
        phif >= quality_criteria["Fair"]["phif_min"]
        and sw <= quality_criteria["Fair"]["sw_max"]
        and vsh <= quality_criteria["Fair"]["vsh_max"]
    ):
        return "Fair"

    else:
        return None


def calculate_reservoir_properties(zone_data):
    """
    Calculate detailed petrophysical properties of a reservoir zone.

    This function computes key reservoir attributes including:
    - Porosity, water saturation, and shale volume statistics
    - Hydrocarbon saturation
    - Net-to-gross ratio
    - Effective reservoir ratio
    - Oil-bearing interval ratio
    - Sampling continuity metrics

    Parameters:
        zone_data (pd.DataFrame): DataFrame containing reservoir zone data,
            expected to include columns:
            - PHIF_pred
            - SW_pred
            - VSH_pred
            - DEPTH

    Returns:
        dict: Dictionary of computed reservoir properties.
    """

    properties = {}

    # =========================================================
    # Basic petrophysical statistics
    # =========================================================
    properties["avg_phif"] = zone_data["PHIF_pred"].mean()
    properties["avg_sw"] = zone_data["SW_pred"].mean()
    properties["avg_vsh"] = zone_data["VSH_pred"].mean()

    # Hydrocarbon saturation
    properties["avg_so"] = 1 - properties["avg_sw"]

    # =========================================================
    # Range statistics
    # =========================================================
    properties["phif_range"] = (
        zone_data["PHIF_pred"].min(),
        zone_data["PHIF_pred"].max()
    )

    properties["sw_range"] = (
        zone_data["SW_pred"].min(),
        zone_data["SW_pred"].max()
    )

    properties["vsh_range"] = (
        zone_data["VSH_pred"].min(),
        zone_data["VSH_pred"].max()
    )

    # =========================================================
    # Data quality metrics
    # =========================================================
    properties["data_points"] = len(zone_data)

    # Net-to-gross ratio (clean sand fraction)
    net_sand_points = zone_data[zone_data["VSH_pred"] < 0.35]
    properties["net_gross_ratio"] = (
        len(net_sand_points) / len(zone_data)
        if len(zone_data) > 0 else 0
    )

    # Effective reservoir ratio
    effective_points = zone_data[
        (zone_data["PHIF_pred"] > 0.1) &
        (zone_data["SW_pred"] < 0.7)
    ]

    properties["effective_ratio"] = (
        len(effective_points) / len(zone_data)
        if len(zone_data) > 0 else 0
    )

    # Oil-bearing interval ratio
    oil_bearing_points = zone_data[zone_data["SW_pred"] < 0.5]

    properties["oil_bearing_ratio"] = (
        len(oil_bearing_points) / len(zone_data)
        if len(zone_data) > 0 else 0
    )

    # =========================================================
    # Continuity metrics
    # =========================================================
    sorted_depth = zone_data["DEPTH"].sort_values()
    depth_intervals = np.diff(sorted_depth)

    properties["avg_sampling_interval"] = (
        depth_intervals.mean() if len(depth_intervals) > 0 else 0
    )

    properties["max_gap"] = (
        depth_intervals.max() if len(depth_intervals) > 0 else 0
    )

    return properties


def identify_reservoir_zones(df, min_thickness=15, max_gap=5, quality_criteria=None):
    """
    Identify potential reservoir zones from well log data using rule-based segmentation.

    This function performs depth-wise segmentation of reservoir intervals based on:
    - Petrophysical quality classification (PHIF, SW, VSH)
    - Continuity constraints (max depth gap)
    - Minimum thickness threshold

    Each identified zone is further enriched with detailed petrophysical properties.

    Parameters:
        df (pd.DataFrame): Well log data containing at least:
            - DEPTH
            - PHIF_pred
            - SW_pred
            - VSH_pred

        min_thickness (float): Minimum reservoir thickness (depth units, e.g., ft).
        max_gap (float): Maximum allowed depth discontinuity within a zone.
        quality_criteria (dict, optional): Reservoir classification thresholds.

    Returns:
        list: List of reservoir zones, each containing:
            - start_depth
            - end_depth
            - thickness
            - quality label
            - petrophysical properties
            - zone statistics
    """

    zones = []
    current_zone = None

    # Sort by depth
    df = df.sort_values("DEPTH")

    # =========================================================
    # Step 1: Sequential zone detection
    # =========================================================
    for _, row in df.iterrows():

        quality = classify_reservoir_quality(
            row["PHIF_pred"],
            row["SW_pred"],
            row["VSH_pred"],
            quality_criteria
        )

        # -----------------------------------------------------
        # Case 1: Reservoir interval detected
        # -----------------------------------------------------
        if quality:

            if current_zone is None:
                # Start new zone
                current_zone = {
                    "start_depth": row["DEPTH"],
                    "end_depth": row["DEPTH"],
                    "points": [row],
                    "quality": quality
                }

            else:
                # Check continuity constraint
                if row["DEPTH"] - current_zone["end_depth"] <= max_gap:

                    current_zone["end_depth"] = row["DEPTH"]
                    current_zone["points"].append(row)

                    # Upgrade quality if necessary
                    if (
                        quality == "Excellent"
                        or (quality == "Good" and current_zone["quality"] == "Fair")
                    ):
                        current_zone["quality"] = quality

                else:
                    # Close current zone
                    if (
                        current_zone["end_depth"]
                        - current_zone["start_depth"]
                        >= min_thickness
                    ):
                        zones.append(current_zone)

                    # Start new zone
                    current_zone = {
                        "start_depth": row["DEPTH"],
                        "end_depth": row["DEPTH"],
                        "points": [row],
                        "quality": quality
                    }

        # -----------------------------------------------------
        # Case 2: Non-reservoir interval
        # -----------------------------------------------------
        else:
            if current_zone is not None:

                if (
                    current_zone["end_depth"]
                    - current_zone["start_depth"]
                    >= min_thickness
                ):
                    zones.append(current_zone)

                current_zone = None

    # =========================================================
    # Step 2: Final zone closure
    # =========================================================
    if (
        current_zone is not None
        and (current_zone["end_depth"] - current_zone["start_depth"] >= min_thickness)
    ):
        zones.append(current_zone)

    # =========================================================
    # Step 3: Compute zone properties
    # =========================================================
    for i, zone in enumerate(zones):

        zone_df = pd.DataFrame(zone["points"])

        zone["number"] = i + 1
        zone["thickness"] = zone["end_depth"] - zone["start_depth"]

        # Compute petrophysical properties
        properties = calculate_reservoir_properties(zone_df)
        zone.update(properties)

    return zones


def calculate_well_summary(well_data, reservoir_zones):
    """
    Compute comprehensive well-level reservoir summary statistics.

    This function aggregates zone-level reservoir information into
    a single well-level geological and petrophysical summary, including:

    - Total depth coverage
    - Reservoir development intensity
    - Reservoir quality distribution
    - Best reservoir zone identification
    - Average petrophysical properties

    Parameters:
        well_data (pd.DataFrame): Full well logging dataset, including:
            - WELLNUM
            - DEPTH
            - PHIF_pred
            - SW_pred
            - VSH_pred

        reservoir_zones (list): List of identified reservoir zones,
            each containing zone-level properties.

    Returns:
        dict: Well-level summary containing:
            - reservoir statistics
            - quality distribution
            - best reservoir zone
            - averaged petrophysical properties
    """

    summary = {}

    # =========================================================
    # Basic well information
    # =========================================================
    summary["well_number"] = well_data["WELLNUM"].iloc[0]

    summary["total_depth_range"] = (
        well_data["DEPTH"].min(),
        well_data["DEPTH"].max()
    )

    summary["total_logged_thickness"] = (
        well_data["DEPTH"].max() - well_data["DEPTH"].min()
    )

    # =========================================================
    # Reservoir statistics
    # =========================================================
    summary["total_reservoir_zones"] = len(reservoir_zones)

    summary["total_reservoir_thickness"] = sum(
        zone["thickness"] for zone in reservoir_zones
    )

    summary["reservoir_density"] = (
        summary["total_reservoir_thickness"]
        / summary["total_logged_thickness"]
        * 100
        if summary["total_logged_thickness"] > 0 else 0
    )

    # =========================================================
    # Reservoir quality distribution
    # =========================================================
    quality_stats = {"Excellent": 0, "Good": 0, "Fair": 0}
    quality_thickness = {"Excellent": 0, "Good": 0, "Fair": 0}

    for zone in reservoir_zones:
        quality = zone["quality"]
        quality_stats[quality] += 1
        quality_thickness[quality] += zone["thickness"]

    summary["quality_distribution"] = quality_stats
    summary["quality_thickness_distribution"] = quality_thickness

    # =========================================================
    # Best reservoir zone selection
    # =========================================================
    if reservoir_zones:

        best_zone = max(
            reservoir_zones,
            key=lambda x: x["avg_so"] * x["avg_phif"] * (1 - x["avg_vsh"])
        )

        summary["best_reservoir"] = {
            "zone_number": best_zone["number"],
            "depth_range": (
                best_zone["start_depth"],
                best_zone["end_depth"]
            ),
            "thickness": best_zone["thickness"],
            "quality": best_zone["quality"],
            "avg_phif": best_zone["avg_phif"],
            "avg_sw": best_zone["avg_sw"],
            "avg_so": best_zone["avg_so"],
            "reservoir_quality_index":
                best_zone["avg_so"]
                * best_zone["avg_phif"]
                * (1 - best_zone["avg_vsh"])
        }

    else:
        summary["best_reservoir"] = None

    # =========================================================
    # Well-level averaged properties
    # =========================================================
    summary["well_avg_phif"] = well_data["PHIF_pred"].mean()
    summary["well_avg_sw"] = well_data["SW_pred"].mean()
    summary["well_avg_vsh"] = well_data["VSH_pred"].mean()

    return summary


def calculate_rmse(well_data):
    """
    Compute Root Mean Square Error (RMSE) between predicted and ground truth petrophysical properties.

    This function evaluates model performance for:
    - Porosity (PHIF)
    - Water saturation (SW)
    - Shale volume (VSH)

    It also computes an averaged RMSE across all properties.

    Parameters:
        well_data (pd.DataFrame): DataFrame containing both ground truth and predicted values:
            - PHIF, PHIF_pred
            - SW, SW_pred
            - VSH, VSH_pred

    Returns:
        tuple:
            - phif_rmse (float): RMSE for porosity
            - sw_rmse (float): RMSE for water saturation
            - vsh_rmse (float): RMSE for shale volume
            - avg_rmse (float): Mean RMSE across all properties
    """

    phif_rmse = np.sqrt(np.mean((well_data["PHIF"] - well_data["PHIF_pred"]) ** 2))
    sw_rmse = np.sqrt(np.mean((well_data["SW"] - well_data["SW_pred"]) ** 2))
    vsh_rmse = np.sqrt(np.mean((well_data["VSH"] - well_data["VSH_pred"]) ** 2))

    avg_rmse = (phif_rmse + sw_rmse + vsh_rmse) / 3

    return phif_rmse, sw_rmse, vsh_rmse, avg_rmse


def setup_figure_and_axes(figsize=(18, 24), width_ratios=[1, 1, 1, 0.6]):
    """
    Create and configure a multi-panel Matplotlib figure for well log visualization.

    This function sets up a standardized figure layout with four aligned subplots:
    - PHIF (porosity)
    - SW (water saturation)
    - VSH (shale volume)
    - Reservoir interpretation panel

    All axes share the same depth axis for consistent geological comparison.

    Parameters:
        figsize (tuple): Figure size in inches (width, height).
        width_ratios (list): Relative widths of the four subplot panels.

    Returns:
        tuple:
            - fig: Matplotlib Figure object
            - ax0: PHIF axis
            - ax1: SW axis
            - ax2: VSH axis
            - ax3: Reservoir interpretation axis
    """

    fig = plt.figure(figsize=figsize)

    # Create grid layout for multi-track well log display
    gs = GridSpec(1, 4, width_ratios=width_ratios, figure=fig)

    # =========================================================
    # Subplots (shared depth axis)
    # =========================================================
    ax0 = fig.add_subplot(gs[0])              # Porosity (PHIF)
    ax1 = fig.add_subplot(gs[1], sharey=ax0)  # Water saturation (SW)
    ax2 = fig.add_subplot(gs[2], sharey=ax0)  # Shale volume (VSH)
    ax3 = fig.add_subplot(gs[3], sharey=ax0)  # Reservoir interpretation panel

    return fig, ax0, ax1, ax2, ax3


def setup_depth_ticks(ax0, ax1, ax2, ax3, min_depth, max_depth, grid_style=None):
    """
    Configure depth axis ticks, grid lines, and formatting for multi-track well log plots.

    This function standardizes the depth axis across multiple subplots, ensuring:
    - Consistent depth scaling (increasing downward)
    - Major and minor tick generation
    - Unified grid styling
    - Clean publication-quality visualization

    Parameters:
        ax0, ax1, ax2, ax3: Matplotlib axis objects for multi-track logging panels.
        min_depth (float): Minimum depth value.
        max_depth (float): Maximum depth value.
        grid_style (dict, optional): Custom grid styling parameters.

    Returns:
        tuple:
            - major_ticks (np.ndarray): Major depth tick locations
            - minor_ticks (np.ndarray): Minor depth tick locations
    """

    # =========================================================
    # Default grid configuration
    # =========================================================
    if grid_style is None:
        grid_style = {
            "major_color": "#777777",
            "major_alpha": 0.8,
            "major_linewidth": 0.8,
            "minor_color": "#999999",
            "minor_alpha": 0.5,
            "minor_linewidth": 0.4
        }

    depth_span = max_depth - min_depth

    # =========================================================
    # Adaptive tick interval selection
    # =========================================================
    if depth_span > 2000:
        depth_interval = 200
    elif depth_span > 1000:
        depth_interval = 100
    else:
        depth_interval = 50

    # =========================================================
    # Generate major and minor ticks
    # =========================================================
    major_ticks = np.arange(
        np.floor(min_depth / depth_interval) * depth_interval,
        np.ceil(max_depth / depth_interval) * depth_interval + 1,
        depth_interval
    )

    minor_interval = depth_interval / 5

    minor_ticks = np.arange(
        np.floor(min_depth / minor_interval) * minor_interval,
        np.ceil(max_depth / minor_interval) * minor_interval + 1,
        minor_interval
    )

    # =========================================================
    # Apply formatting to all axes
    # =========================================================
    for ax in [ax0, ax1, ax2, ax3]:

        # Depth increases downward (standard well log convention)
        ax.set_ylim(max_depth, min_depth)

        # Set ticks
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)

        # Major grid
        ax.grid(
            True,
            which="major",
            linestyle="-",
            linewidth=grid_style["major_linewidth"],
            color=grid_style["major_color"],
            alpha=grid_style["major_alpha"]
        )

        # Minor grid
        ax.grid(
            True,
            which="minor",
            linestyle=":",
            linewidth=grid_style["minor_linewidth"],
            color=grid_style["minor_color"],
            alpha=grid_style["minor_alpha"]
        )

        # Axis styling
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("black")

        # =====================================================
        # Left panel (main depth label)
        # =====================================================
        if ax == ax0:
            ax.set_ylabel("Depth (ft)", fontsize=12, fontweight="bold")
            ax.tick_params(axis="y", which="major", labelsize=10)

            for tick in major_ticks:
                ax.text(
                    -0.05,
                    tick,
                    f"{int(tick)}",
                    ha="right",
                    va="center",
                    fontsize=9,
                    transform=ax.get_yaxis_transform(),
                    bbox=dict(facecolor="white", edgecolor="none", pad=0)
                )

        # =====================================================
        # Right panel (reservoir interpretation panel)
        # =====================================================
        elif ax == ax3:
            ax.tick_params(axis="y", which="major", labelsize=10)

            for tick in major_ticks:
                ax.text(
                    1.05,
                    tick,
                    f"{int(tick)}",
                    ha="left",
                    va="center",
                    fontsize=9,
                    transform=ax.get_yaxis_transform(),
                    bbox=dict(facecolor="white", edgecolor="none", pad=0)
                )

            ax.set_yticklabels([])

        # =====================================================
        # Middle panels (no y-labels)
        # =====================================================
        else:
            ax.set_yticklabels([])

    return major_ticks, minor_ticks


def draw_reservoir_zones(ax0, ax1, ax2, reservoir_zones, color_scheme=None):
    """
    Visualize interpreted reservoir zones on multi-track well log plots.

    This function overlays reservoir intervals onto PHIF, SW, and VSH tracks,
    using color-coded backgrounds based on reservoir quality classification.

    Each zone is represented by:
    - A shaded interval (zone body)
    - Boundary lines (top and bottom)
    - Depth annotations (on PHIF track only)

    Parameters:
        ax0, ax1, ax2: Matplotlib axis objects corresponding to:
            - PHIF (porosity)
            - SW (water saturation)
            - VSH (shale volume)

        reservoir_zones (list): List of reservoir zone dictionaries, each containing:
            - start_depth
            - end_depth
            - thickness
            - quality (Excellent / Good / Fair)

        color_scheme (dict, optional): Custom color mapping for reservoir quality.

    Returns:
        None
    """

    # =========================================================
    # Default color scheme
    # =========================================================
    if color_scheme is None:
        color_scheme = {
            "Excellent": {
                "bg_color": "#b8e0b8",
                "edge_color": "#006400"
            },
            "Good": {
                "bg_color": "#cce8cc",
                "edge_color": "#228B22"
            },
            "Fair": {
                "bg_color": "#e0f0e0",
                "edge_color": "#3CB371"
            }
        }

    # =========================================================
    # Draw each reservoir zone
    # =========================================================
    for zone in reservoir_zones:

        quality = zone["quality"]

        bg_color = color_scheme[quality]["bg_color"]
        edge_color = color_scheme[quality]["edge_color"]

        # Apply to all logging tracks
        for axis in [ax0, ax1, ax2]:

            # -------------------------------------------------
            # Zone shading
            # -------------------------------------------------
            rect = patches.Rectangle(
                (axis.get_xlim()[0], zone["start_depth"]),
                axis.get_xlim()[1] - axis.get_xlim()[0],
                zone["thickness"],
                facecolor=bg_color,
                edgecolor=edge_color,
                linewidth=0.8,
                zorder=1
            )
            axis.add_patch(rect)

            # -------------------------------------------------
            # Zone boundaries
            # -------------------------------------------------
            axis.axhline(
                y=zone["start_depth"],
                color=edge_color,
                linestyle="-",
                linewidth=1.0
            )

            axis.axhline(
                y=zone["end_depth"],
                color=edge_color,
                linestyle="-",
                linewidth=1.0
            )

            # -------------------------------------------------
            # Depth annotations (only PHIF track)
            # -------------------------------------------------
            if axis == ax0:

                ax0.text(
                    -0.05,
                    zone["start_depth"],
                    f"{int(zone['start_depth'])}",
                    ha="right",
                    va="bottom",
                    fontsize=8,
                    color=edge_color,
                    weight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0)
                )

                ax0.text(
                    -0.05,
                    zone["end_depth"],
                    f"{int(zone['end_depth'])}",
                    ha="right",
                    va="top",
                    fontsize=8,
                    color=edge_color,
                    weight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0)
                )


def setup_parameter_axes(ax0, ax1, ax2, phif_max=0.4):
    """
    Configure x-axis settings for petrophysical parameter tracks in well log plots.

    This function standardizes the horizontal axis for three key logging curves:
    - Effective Porosity (PHIF)
    - Water Saturation (SW)
    - Shale Volume (VSH)

    It sets:
    - Axis limits
    - Major and minor ticks
    - Tick formatting
    - Axis labels

    Parameters:
        ax0, ax1, ax2: Matplotlib axis objects corresponding to:
            - PHIF track
            - SW track
            - VSH track

        phif_max (float): Maximum value for porosity axis (PHIF).

    Returns:
        None
    """

    for ax, title, param in zip(
        [ax0, ax1, ax2],
        ["Effective Porosity (PHIF)", "Water Saturation (SW)", "Shale Volume (VSH)"],
        ["PHIF", "SW", "VSH"]
    ):

        # =========================================================
        # X-axis range and major ticks
        # =========================================================
        if param == "PHIF":
            xlim = (0, phif_max)
            xticks = np.linspace(0, phif_max, 5)
        else:
            xlim = (0, 1.0)
            xticks = np.linspace(0, 1.0, 6)

        ax.set_xlim(xlim)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x:.1f}" for x in xticks])
        ax.tick_params(axis="x", which="major", labelsize=9)

        # =========================================================
        # Minor ticks (for grid refinement)
        # =========================================================
        if param == "PHIF":
            minor_xticks = np.linspace(0, phif_max, 21)[1:-1:2]
        else:
            minor_xticks = np.linspace(0, 1.0, 21)[1:-1:2]

        ax.set_xticks(minor_xticks, minor=True)

        # =========================================================
        # Axis label styling
        # =========================================================
        ax.set_xlabel(param, fontsize=10, fontweight="bold")


def plot_curves(ax0, ax1, ax2, well_data, phif_rmse, sw_rmse, vsh_rmse, curve_style=None):
    """
    Plot actual vs predicted petrophysical curves for well log analysis.

    This function visualizes:
    - Effective porosity (PHIF)
    - Water saturation (SW)
    - Shale volume (VSH)

    It compares ground truth and model predictions along depth and annotates
    each track with RMSE values.

    Parameters:
        ax0, ax1, ax2: Matplotlib axes for:
            - PHIF track
            - SW track
            - VSH track

        well_data (pd.DataFrame): Well logging data containing:
            - PHIF, PHIF_pred
            - SW, SW_pred
            - VSH, VSH_pred
            - DEPTH

        phif_rmse (float): RMSE of porosity prediction.
        sw_rmse (float): RMSE of water saturation prediction.
        vsh_rmse (float): RMSE of shale volume prediction.

        curve_style (dict, optional): Styling configuration for curves.

    Returns:
        None
    """

    # =========================================================
    # Default curve styling
    # =========================================================
    if curve_style is None:
        curve_style = {
            "actual_color": "blue",
            "actual_linewidth": 1.5,
            "predicted_color": "red",
            "predicted_linewidth": 1.2,
            "predicted_linestyle": "--"
        }

    # =========================================================
    # PHIF track
    # =========================================================
    ax0.set_title(
        f"Effective Porosity (PHIF)\nRMSE: {phif_rmse:.4f}",
        fontsize=11,
        fontweight="bold"
    )

    ax0.plot(
        well_data["PHIF"],
        well_data["DEPTH"],
        "-",
        color=curve_style["actual_color"],
        linewidth=curve_style["actual_linewidth"],
        label="Actual",
        zorder=2
    )

    ax0.plot(
        well_data["PHIF_pred"],
        well_data["DEPTH"],
        curve_style["predicted_linestyle"],
        color=curve_style["predicted_color"],
        linewidth=curve_style["predicted_linewidth"],
        label="Predicted",
        zorder=2
    )

    ax0.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # =========================================================
    # SW track
    # =========================================================
    ax1.set_title(
        f"Water Saturation (SW)\nRMSE: {sw_rmse:.4f}",
        fontsize=11,
        fontweight="bold"
    )

    ax1.plot(
        well_data["SW"],
        well_data["DEPTH"],
        "-",
        color=curve_style["actual_color"],
        linewidth=curve_style["actual_linewidth"],
        label="Actual",
        zorder=2
    )

    ax1.plot(
        well_data["SW_pred"],
        well_data["DEPTH"],
        curve_style["predicted_linestyle"],
        color=curve_style["predicted_color"],
        linewidth=curve_style["predicted_linewidth"],
        label="Predicted",
        zorder=2
    )

    ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # =========================================================
    # VSH track
    # =========================================================
    ax2.set_title(
        f"Shale Volume (VSH)\nRMSE: {vsh_rmse:.4f}",
        fontsize=11,
        fontweight="bold"
    )

    ax2.plot(
        well_data["VSH"],
        well_data["DEPTH"],
        "-",
        color=curve_style["actual_color"],
        linewidth=curve_style["actual_linewidth"],
        label="Actual",
        zorder=2
    )

    ax2.plot(
        well_data["VSH_pred"],
        well_data["DEPTH"],
        curve_style["predicted_linestyle"],
        color=curve_style["predicted_color"],
        linewidth=curve_style["predicted_linewidth"],
        label="Predicted",
        zorder=2
    )

    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)


def setup_analysis_panel(ax3, reservoir_zones, fig, quality_criteria=None):
    """
    Create and configure the reservoir interpretation panel for well-log visualization.

    This panel provides:
    - Zone-level reservoir summaries
    - Petrophysical property display
    - Quality classification visualization
    - Reference criteria table

    The layout automatically adjusts based on the number of reservoir zones
    to ensure readability and publication-quality presentation.

    Parameters:
        ax3: Matplotlib axis for reservoir interpretation panel.
        reservoir_zones (list): List of reservoir zones with properties:
            - start_depth
            - end_depth
            - thickness
            - avg_phif
            - avg_sw
            - avg_vsh
            - avg_so
            - net_gross_ratio
            - quality

        fig: Matplotlib figure object (used for text sizing).
        quality_criteria (dict, optional): Reservoir classification thresholds.

    Returns:
        None
    """

    # =========================================================
    # Panel base configuration
    # =========================================================
    ax3.set_title("Reservoir Analysis", fontsize=11, fontweight="bold")
    ax3.set_xlim(0, 1)
    ax3.set_xticks([])
    ax3.set_facecolor("white")
    ax3.grid(False)

    if len(reservoir_zones) == 0:
        return

    # =========================================================
    # Layout configuration (adaptive spacing)
    # =========================================================
    available_height = 0.75
    start_y = 0.95

    if len(reservoir_zones) > 10:
        fontsize = 7
    elif len(reservoir_zones) > 5:
        fontsize = 8
    else:
        fontsize = 9

    positions = np.linspace(
        start_y,
        start_y - available_height,
        len(reservoir_zones)
    )

    # =========================================================
    # Zone rendering
    # =========================================================
    for i, zone in enumerate(reservoir_zones):

        y_pos = positions[i]

        # -----------------------------------------------------
        # Color mapping by reservoir quality
        # -----------------------------------------------------
        if zone["quality"] == "Excellent":
            bg_color = "#b8e0b8"
            edge_color = "#006400"
        elif zone["quality"] == "Good":
            bg_color = "#cce8cc"
            edge_color = "#228B22"
        else:
            bg_color = "#e0f0e0"
            edge_color = "#3CB371"

        # -----------------------------------------------------
        # Zone information text
        # -----------------------------------------------------
        zone_info = (
            f"Reservoir {i + 1}: {int(zone['start_depth'])}-{int(zone['end_depth'])} ft\n"
            f"Thickness: {zone['thickness']:.1f} ft\n"
            f"PHIF: {zone['avg_phif']:.3f} | SW: {zone['avg_sw']:.3f}\n"
            f"VSH: {zone['avg_vsh']:.3f} | SO: {zone['avg_so']:.3f}\n"
            f"Net/Gross: {zone['net_gross_ratio']:.2f}\n"
            f"Quality: {zone['quality']}"
        )

        # -----------------------------------------------------
        # Try dynamic text bounding box
        # -----------------------------------------------------
        try:
            text_obj = ax3.text(0, 0, zone_info, fontsize=fontsize, linespacing=1.1)
            renderer = fig.canvas.get_renderer()

            bbox = text_obj.get_window_extent(renderer=renderer).transformed(
                ax3.transAxes.inverted()
            )

            text_width = bbox.width * 1.1
            text_height = bbox.height * 1.1
            text_obj.remove()

            rect = patches.Rectangle(
                (0.5 - text_width / 2, y_pos - text_height / 2),
                text_width,
                text_height,
                transform=ax3.transAxes,
                facecolor=bg_color,
                edgecolor=edge_color,
                linewidth=0.8,
                zorder=3,
                alpha=0.9
            )

        except:
            # Fallback fixed-size box
            rect = patches.Rectangle(
                (0.1, y_pos - 0.07),
                0.8,
                0.14,
                transform=ax3.transAxes,
                facecolor=bg_color,
                edgecolor=edge_color,
                linewidth=0.8,
                zorder=3,
                alpha=0.9
            )

        ax3.add_patch(rect)

        ax3.text(
            0.5,
            y_pos,
            zone_info,
            transform=ax3.transAxes,
            ha="center",
            va="center",
            fontsize=fontsize,
            linespacing=1.1,
            weight="bold"
        )

    # =========================================================
    # Quality criteria panel (bottom reference box)
    # =========================================================
    if quality_criteria is None:
        quality_text = (
            "Reservoir Quality Criteria:\n"
            "Excellent: PHIF≥0.20, SW≤0.30, VSH≤0.20\n"
            "Good: PHIF≥0.15, SW≤0.40, VSH≤0.30\n"
            "Fair: PHIF≥0.10, SW≤0.50, VSH≤0.35"
        )
    else:
        quality_text = (
            "Reservoir Quality Criteria:\n"
            f"Excellent: PHIF≥{quality_criteria['Excellent']['phif_min']:.2f}, "
            f"SW≤{quality_criteria['Excellent']['sw_max']:.2f}, "
            f"VSH≤{quality_criteria['Excellent']['vsh_max']:.2f}\n"
            f"Good: PHIF≥{quality_criteria['Good']['phif_min']:.2f}, "
            f"SW≤{quality_criteria['Good']['sw_max']:.2f}, "
            f"VSH≤{quality_criteria['Good']['vsh_max']:.2f}\n"
            f"Fair: PHIF≥{quality_criteria['Fair']['phif_min']:.2f}, "
            f"SW≤{quality_criteria['Fair']['sw_max']:.2f}, "
            f"VSH≤{quality_criteria['Fair']['vsh_max']:.2f}"
        )

    # =========================================================
    # Render quality reference box
    # =========================================================
    try:
        text_obj = ax3.text(0, 0, quality_text, fontsize=8, linespacing=1.2)
        renderer = fig.canvas.get_renderer()

        bbox = text_obj.get_window_extent(renderer=renderer).transformed(
            ax3.transAxes.inverted()
        )

        text_width = bbox.width * 1.1
        text_height = bbox.height * 1.1
        text_obj.remove()

        quality_box = patches.Rectangle(
            (0.5 - text_width / 2, 0.05 - text_height / 2),
            text_width,
            text_height,
            transform=ax3.transAxes,
            facecolor="#f9f9f9",
            edgecolor="black",
            linewidth=0.8,
            zorder=4
        )

    except:
        quality_box = patches.Rectangle(
            (0.1, 0.02),
            0.8,
            0.1,
            transform=ax3.transAxes,
            facecolor="#f9f9f9",
            edgecolor="black",
            linewidth=0.8,
            zorder=4
        )

    ax3.add_patch(quality_box)

    ax3.text(
        0.5,
        0.05,
        quality_text,
        transform=ax3.transAxes,
        ha="center",
        va="center",
        fontsize=8,
        linespacing=1.2,
        zorder=5
    )


def add_summary_box(fig, avg_rmse, reservoir_zones):
    """
    Add a summary information box at the bottom of the figure.

    This function creates a global annotation panel that summarizes:
    - Model performance (average RMSE)
    - Reservoir detection results (number of zones)
    - Total reservoir thickness

    The summary box is rendered in figure coordinates to ensure
    consistent placement across different subplot layouts.

    Parameters:
        fig: Matplotlib figure object.
        avg_rmse (float): Average RMSE of model predictions.
        reservoir_zones (list): List of detected reservoir zones.

    Returns:
        None
    """

    # =========================================================
    # Summary background box
    # =========================================================
    summary_box = patches.Rectangle(
        (0.0, 0.0),
        1.0,
        0.04,
        transform=fig.transFigure,
        facecolor="#f0f0f0",
        edgecolor="black",
        linewidth=0.8,
        zorder=3
    )

    fig.patches.append(summary_box)

    # =========================================================
    # Compute summary metrics
    # =========================================================
    total_thickness = (
        sum(zone["thickness"] for zone in reservoir_zones)
        if reservoir_zones else 0
    )

    summary_text = (
        f"Average RMSE = {avg_rmse:.4f}    "
        f"Identified {len(reservoir_zones)} reservoir zones    "
        f"Total reservoir thickness = {total_thickness:.0f} ft"
    )

    # =========================================================
    # Render summary text
    # =========================================================
    fig.text(
        0.5,
        0.02,
        summary_text,
        ha="center",
        va="center",
        fontsize=10,
        weight="bold"
    )


def generate_reservoir_report(well_summary, reservoir_zones):
    """
    Generate a comprehensive reservoir evaluation report for a single well.

    This function performs structured reporting of:
    - Well-level reservoir summary statistics
    - Quality distribution analysis
    - Best reservoir zone identification
    - Detailed zone-by-zone interpretation
    - Final geological and engineering recommendations

    Robust error handling is included to ensure missing fields do not break execution.

    Parameters:
        well_summary (dict): Well-level reservoir summary information.
        reservoir_zones (list): List of reservoir zone dictionaries.

    Returns:
        str: Formatted reservoir evaluation report.
    """

    # =========================================================
    # Field validation utilities
    # =========================================================
    def ensure_zone_fields(zone):
        """Ensure all required reservoir zone fields exist."""
        required_fields = {
            "avg_phif": 0.0,
            "avg_sw": 0.0,
            "avg_vsh": 0.0,
            "avg_so": 0.0,
            "thickness": 0.0,
            "start_depth": 0.0,
            "end_depth": 0.0,
            "quality": "Unknown",
            "number": 0,
            "phif_range": (0.0, 0.0),
            "sw_range": (0.0, 0.0),
            "vsh_range": (0.0, 0.0),
            "net_gross_ratio": 0.0,
            "effective_ratio": 0.0,
            "oil_bearing_ratio": 0.0,
            "data_points": 0,
            "avg_sampling_interval": 0.0,
            "max_gap": 0.0,
        }

        for field, default in required_fields.items():
            if field not in zone or zone[field] is None:
                zone[field] = default

        return zone

    def ensure_summary_fields(summary):
        """Ensure all required well summary fields exist."""
        required_fields = {
            "well_number": "Unknown",
            "total_depth_range": (0.0, 0.0),
            "total_logged_thickness": 0.0,
            "total_reservoir_zones": 0,
            "total_reservoir_thickness": 0.0,
            "reservoir_density": 0.0,
            "well_avg_phif": 0.0,
            "well_avg_sw": 0.0,
            "well_avg_vsh": 0.0,
            "quality_distribution": {"Excellent": 0, "Good": 0, "Fair": 0},
            "quality_thickness_distribution": {"Excellent": 0, "Good": 0, "Fair": 0},
            "best_reservoir": None,
        }

        for field, default in required_fields.items():
            if field not in summary or summary[field] is None:
                summary[field] = default

        return summary

    # =========================================================
    # Data validation
    # =========================================================
    try:
        well_summary = ensure_summary_fields(well_summary)
        reservoir_zones = [ensure_zone_fields(z) for z in reservoir_zones]
    except Exception as e:
        return f"Report generation failed: data validation error - {str(e)}"

    # =========================================================
    # Report container
    # =========================================================
    report = []

    try:
        # =====================================================
        # Header
        # =====================================================
        report.append("=" * 80)
        report.append(f"RESERVOIR EVALUATION REPORT - WELL {well_summary['well_number']}")
        report.append("=" * 80)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # =====================================================
        # 1. Well overview
        # =====================================================
        report.append("1. WELL OVERVIEW")
        report.append("-" * 40)

        report.append(f"Well Number: {well_summary['well_number']}")

        depth_range = well_summary["total_depth_range"]
        if hasattr(depth_range, "__len__") and len(depth_range) == 2:
            report.append(
                f"Depth Range: {float(depth_range[0]):.1f} - {float(depth_range[1]):.1f} ft"
            )
        else:
            report.append("Depth Range: N/A")

        report.append(f"Total Logged Thickness: {float(well_summary['total_logged_thickness']):.1f} ft")

        report.append("Average Well Properties:")
        report.append(f"  - PHIF: {float(well_summary['well_avg_phif']):.3f}")
        report.append(f"  - SW: {float(well_summary['well_avg_sw']):.3f}")
        report.append(f"  - VSH: {float(well_summary['well_avg_vsh']):.3f}")
        report.append("")

        # =====================================================
        # 2. Reservoir summary
        # =====================================================
        report.append("2. RESERVOIR SUMMARY")
        report.append("-" * 40)

        report.append(f"Total Reservoir Zones: {int(well_summary['total_reservoir_zones'])}")
        report.append(f"Total Reservoir Thickness: {float(well_summary['total_reservoir_thickness']):.1f} ft")
        report.append(f"Reservoir Density: {float(well_summary['reservoir_density']):.1f}%")
        report.append("")

        # =====================================================
        # 3. Quality distribution
        # =====================================================
        report.append("3. RESERVOIR QUALITY DISTRIBUTION")
        report.append("-" * 40)

        quality_stats = well_summary["quality_distribution"]
        quality_thickness = well_summary["quality_thickness_distribution"]

        for q in ["Excellent", "Good", "Fair"]:
            count = int(quality_stats.get(q, 0))
            thick = float(quality_thickness.get(q, 0))

            if count > 0:
                report.append(
                    f"{q}: {count} zones, {thick:.1f} ft total"
                )

        report.append("")

        # =====================================================
        # 4. Best reservoir zone
        # =====================================================
        if well_summary["best_reservoir"]:
            best = well_summary["best_reservoir"]

            report.append("4. BEST RESERVOIR ZONE")
            report.append("-" * 40)

            report.append(f"Zone: R{int(best.get('zone_number', 0))}")
            report.append(f"Thickness: {float(best.get('thickness', 0)):.1f} ft")
            report.append(f"Quality: {best.get('quality', 'Unknown')}")

            report.append("Properties:")
            report.append(f"  - PHIF: {float(best.get('avg_phif', 0)):.3f}")
            report.append(f"  - SW: {float(best.get('avg_sw', 0)):.3f}")
            report.append(f"  - SO: {float(best.get('avg_so', 0)):.3f}")
            report.append(f"  - VSH: {float(best.get('avg_vsh', 0)):.3f}")

            report.append("")

        # =====================================================
        # 5. Detailed zones
        # =====================================================
        report.append("5. DETAILED ZONE ANALYSIS")
        report.append("-" * 40)

        for i, zone in enumerate(reservoir_zones, 1):
            report.append(f"Zone R{i}:")
            report.append(
                f"  Depth: {float(zone['start_depth']):.1f} - {float(zone['end_depth']):.1f} ft"
            )
            report.append(f"  Thickness: {float(zone['thickness']):.1f} ft")
            report.append(f"  Quality: {zone['quality']}")
            report.append(f"  PHIF: {float(zone['avg_phif']):.3f}")
            report.append(f"  SW: {float(zone['avg_sw']):.3f}")
            report.append(f"  VSH: {float(zone['avg_vsh']):.3f}")
            report.append("")

        # =====================================================
        # 6. Conclusions
        # =====================================================
        report.append("6. CONCLUSIONS")
        report.append("-" * 40)

        n_zones = int(well_summary["total_reservoir_zones"])

        if n_zones == 0:
            report.append("No reservoir zones identified.")
        elif n_zones <= 2:
            report.append("Limited reservoir development.")
        else:
            report.append("Good reservoir development potential.")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    except Exception as e:
        return f"Report generation failed: {str(e)}"


def export_reservoir_data(well_summary, reservoir_zones, output_folder, well_number):
    """
    Export reservoir analysis results into structured files (JSON, CSV, TXT).

    This function performs complete data export for a single well, including:
    - Well-level reservoir summary (JSON)
    - Zone-level reservoir properties (CSV)
    - Human-readable reservoir interpretation report (TXT)

    All outputs are saved in the specified directory with robust error handling
    and NumPy-to-Python type conversion for serialization compatibility.

    Parameters:
        well_summary (dict): Well-level reservoir summary statistics.
        reservoir_zones (list): List of reservoir zone dictionaries.
        output_folder (str): Directory to save exported files.
        well_number (str or int): Identifier of the well.

    Returns:
        dict: Dictionary containing paths of exported files:
            - summary_json
            - zones_csv
            - report_txt
    """

    exported_files = {}

    # =========================================================
    # Ensure output directory exists
    # =========================================================
    os.makedirs(output_folder, exist_ok=True)

    # =========================================================
    # 1. Export well summary (JSON)
    # =========================================================
    try:
        summary_file = os.path.join(
            output_folder,
            f"Well_{well_number}_reservoir_summary.json"
        )

        summary_serializable = convert_numpy_types(well_summary)

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_serializable, f, indent=2, ensure_ascii=False)

        exported_files["summary_json"] = summary_file
        print(f"✔ Well summary exported: {summary_file}")

    except Exception as e:
        print(f"⚠ Failed to export summary JSON: {str(e)}")
        exported_files["summary_json"] = None

    # =========================================================
    # 2. Export reservoir zones (CSV)
    # =========================================================
    if reservoir_zones:

        try:
            zones_data = []

            for zone in reservoir_zones:

                def safe_get(z, key, default=0.0):
                    try:
                        value = z.get(key, default)

                        if isinstance(value, (list, tuple)) and len(value) >= 2:
                            return float(value[0])

                        return float(value) if value is not None else default

                    except (ValueError, TypeError, IndexError):
                        return default

                zones_data.append({
                    "Zone_Number": int(zone.get("number", 0)),
                    "Start_Depth_ft": safe_get(zone, "start_depth"),
                    "End_Depth_ft": safe_get(zone, "end_depth"),
                    "Thickness_ft": safe_get(zone, "thickness"),
                    "Quality": str(zone.get("quality", "Unknown")),

                    "Avg_PHIF": safe_get(zone, "avg_phif"),
                    "Avg_SW": safe_get(zone, "avg_sw"),
                    "Avg_SO": safe_get(zone, "avg_so"),
                    "Avg_VSH": safe_get(zone, "avg_vsh"),

                    "Net_Gross_Ratio": safe_get(zone, "net_gross_ratio"),
                    "Effective_Ratio": safe_get(zone, "effective_ratio"),
                    "Oil_Bearing_Ratio": safe_get(zone, "oil_bearing_ratio"),

                    "Data_Points": int(zone.get("data_points", 0)),
                    "Avg_Sampling_Interval_ft": safe_get(zone, "avg_sampling_interval"),
                    "Max_Gap_ft": safe_get(zone, "max_gap"),
                })

            zones_df = pd.DataFrame(zones_data)

            zones_file = os.path.join(
                output_folder,
                f"Well_{well_number}_reservoir_zones.csv"
            )

            zones_df.to_csv(zones_file, index=False)

            exported_files["zones_csv"] = zones_file
            print(f"✔ Reservoir zones exported: {zones_file}")

        except Exception as e:
            print(f"⚠ Failed to export zones CSV: {str(e)}")
            exported_files["zones_csv"] = None

    # =========================================================
    # 3. Export textual report
    # =========================================================
    try:
        zones_serializable = convert_numpy_types(reservoir_zones)
        summary_serializable = convert_numpy_types(well_summary)

        report_text = generate_reservoir_report(
            summary_serializable,
            zones_serializable
        )

        if "报告生成失败" in report_text:
            print(f"⚠ Report generation failed")
            exported_files["report_txt"] = None

        else:
            report_file = os.path.join(
                output_folder,
                f"Well_{well_number}_reservoir_report.txt"
            )

            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report_text)

            exported_files["report_txt"] = report_file
            print(f"✔ Reservoir report exported: {report_file}")

    except Exception as e:
        print(f"⚠ Failed to export report: {str(e)}")
        exported_files["report_txt"] = None

    return exported_files


def plot_well_predictions(
    model_predictions_path,
    real_test_result_path,
    output_folder,
    options=None
):
    """
    Full pipeline for well-log interpretation, reservoir identification, and visualization.

    This function integrates:
    - Data loading and merging
    - Reservoir zone identification
    - Petrophysical curve visualization
    - Zone-level statistical analysis
    - Reservoir interpretation panel rendering
    - Multi-format data export (JSON/CSV/TXT/PDF)
    - RMSE-based model evaluation

    The output includes both graphical results and structured analytical data
    for each well, enabling reproducible reservoir characterization.

    Parameters:
        model_predictions_path (str): Path to predicted log data (CSV).
        real_test_result_path (str): Path to ground-truth log data (CSV).
        output_folder (str): Directory for saving outputs.
        options (dict, optional): Configuration dictionary controlling:
            - plotting style
            - reservoir thresholds
            - export options
            - visualization parameters

    Returns:
        dict: Comprehensive analysis results containing:
            - analysis_summary (global statistics)
            - wells (per-well interpretation results)
    """

    # =========================================================
    # Setup plotting environment
    # =========================================================
    setup_plotting_environment()

    # =========================================================
    # Default configuration
    # =========================================================
    default_options = {
        "figsize": (18, 24),
        "width_ratios": [1, 1, 1, 0.6],
        "min_thickness": 15,
        "max_gap": 5,
        "export_data": True,
        "dpi": 200,

        "quality_criteria": {
            "Excellent": {"phif_min": 0.20, "sw_max": 0.30, "vsh_max": 0.20},
            "Good": {"phif_min": 0.15, "sw_max": 0.40, "vsh_max": 0.30},
            "Fair": {"phif_min": 0.10, "sw_max": 0.50, "vsh_max": 0.35}
        },

        "grid_style": {
            "major_color": "#777777",
            "major_alpha": 0.8,
            "major_linewidth": 0.8,
            "minor_color": "#999999",
            "minor_alpha": 0.5,
            "minor_linewidth": 0.4
        },

        "color_scheme": {
            "Excellent": {"bg_color": "#b8e0b8", "edge_color": "#006400"},
            "Good": {"bg_color": "#cce8cc", "edge_color": "#228B22"},
            "Fair": {"bg_color": "#e0f0e0", "edge_color": "#3CB371"}
        },

        "curve_style": {
            "actual_color": "blue",
            "actual_linewidth": 1.5,
            "predicted_color": "red",
            "predicted_linewidth": 1.2,
            "predicted_linestyle": "--"
        }
    }

    # Override defaults
    if options:
        for k, v in options.items():
            if isinstance(v, dict) and k in default_options:
                default_options[k].update(v)
            else:
                default_options[k] = v

    # =========================================================
    # Prepare output directory
    # =========================================================
    os.makedirs(output_folder, exist_ok=True)

    # =========================================================
    # Load data
    # =========================================================
    merged_df = load_and_merge_data(
        model_predictions_path,
        real_test_result_path
    )

    well_numbers = merged_df["WELLNUM"].unique()

    # =========================================================
    # Global result container
    # =========================================================
    analysis_results = {
        "analysis_summary": {
            "total_wells": len(well_numbers),
            "analysis_date": datetime.now().isoformat(),
            "output_folder": output_folder,
            "model_predictions_file": model_predictions_path,
            "real_test_file": real_test_result_path,
        },
        "wells": {}
    }

    # =========================================================
    # Per-well processing loop
    # =========================================================
    for well in well_numbers:

        print(f"Processing Well {well}...")

        well_data = merged_df[merged_df["WELLNUM"] == well]

        min_depth = well_data["DEPTH"].min()
        max_depth = well_data["DEPTH"].max()

        # -----------------------------------------------------
        # Reservoir detection
        # -----------------------------------------------------
        reservoir_zones = identify_reservoir_zones(
            well_data,
            min_thickness=default_options["min_thickness"],
            max_gap=default_options["max_gap"],
            quality_criteria=default_options["quality_criteria"]
        )

        reservoir_zones.sort(key=lambda x: x["start_depth"])

        # -----------------------------------------------------
        # Well summary statistics
        # -----------------------------------------------------
        well_summary = calculate_well_summary(
            well_data,
            reservoir_zones
        )

        # -----------------------------------------------------
        # RMSE evaluation
        # -----------------------------------------------------
        phif_rmse, sw_rmse, vsh_rmse, avg_rmse = calculate_rmse(well_data)

        rmse_metrics = {
            "phif_rmse": phif_rmse,
            "sw_rmse": sw_rmse,
            "vsh_rmse": vsh_rmse,
            "avg_rmse": avg_rmse
        }

        # -----------------------------------------------------
        # Figure setup
        # -----------------------------------------------------
        fig, ax0, ax1, ax2, ax3 = setup_figure_and_axes(
            figsize=default_options["figsize"],
            width_ratios=default_options["width_ratios"]
        )

        fig.suptitle(
            f"Well {well} Analysis ({int(min_depth)}-{int(max_depth)} ft)",
            fontsize=16,
            fontweight="bold"
        )

        # -----------------------------------------------------
        # Depth axis setup
        # -----------------------------------------------------
        setup_depth_ticks(
            ax0, ax1, ax2, ax3,
            min_depth, max_depth,
            grid_style=default_options["grid_style"]
        )

        # -----------------------------------------------------
        # Visualization pipeline
        # -----------------------------------------------------
        draw_reservoir_zones(
            ax0, ax1, ax2,
            reservoir_zones,
            color_scheme=default_options["color_scheme"]
        )

        setup_parameter_axes(ax0, ax1, ax2)

        plot_curves(
            ax0, ax1, ax2,
            well_data,
            phif_rmse, sw_rmse, vsh_rmse,
            curve_style=default_options["curve_style"]
        )

        setup_analysis_panel(
            ax3,
            reservoir_zones,
            fig,
            quality_criteria=default_options["quality_criteria"]
        )

        add_summary_box(fig, avg_rmse, reservoir_zones)

        # -----------------------------------------------------
        # Save outputs
        # -----------------------------------------------------
        plt.tight_layout(rect=[0, 0.04, 1, 0.96])

        image_path = os.path.join(output_folder, f"Well_{well}_comparison.png")
        pdf_path = os.path.join(output_folder, f"Well_{well}_comparison.pdf")

        plt.savefig(image_path, dpi=default_options["dpi"], bbox_inches="tight")
        plt.savefig(pdf_path, bbox_inches="tight")
        plt.close()

        # -----------------------------------------------------
        # Export structured data
        # -----------------------------------------------------
        exported_files = {}

        if default_options["export_data"]:
            exported_files = export_reservoir_data(
                well_summary,
                reservoir_zones,
                output_folder,
                well
            )

        # -----------------------------------------------------
        # Store results
        # -----------------------------------------------------
        analysis_results["wells"][str(well)] = {
            "summary": well_summary,
            "reservoir_zones": reservoir_zones,
            "exported_files": exported_files,
            "image_path": image_path,
            "rmse_metrics": rmse_metrics
        }

        print(f"Well {well} completed.")

    # =========================================================
    # Global summary aggregation (FIXED LOGIC)
    # =========================================================
    total_zones = sum(
        len(v["reservoir_zones"])
        for v in analysis_results["wells"].values()
    )

    total_thickness = sum(
        sum(z["thickness"] for z in v["reservoir_zones"])
        for v in analysis_results["wells"].values()
    )

    num_wells = len(well_numbers)

    analysis_results["analysis_summary"].update({
        "total_reservoir_zones": total_zones,
        "total_reservoir_thickness": total_thickness,
        "average_zones_per_well": total_zones / num_wells if num_wells else 0,
        "wells_with_reservoirs": sum(
            1 for v in analysis_results["wells"].values()
            if v["reservoir_zones"]
        )
    })

    # =========================================================
    # Save global results
    # =========================================================
    try:
        overall_file = os.path.join(output_folder, "overall_analysis.json")

        with open(overall_file, "w", encoding="utf-8") as f:
            json.dump(convert_numpy_types(analysis_results), f, indent=2)

        analysis_results["overall_summary_file"] = overall_file

    except Exception as e:
        print(f"Failed to save global results: {e}")
        analysis_results["overall_summary_file"] = None

    # =========================================================
    # Final print
    # =========================================================
    print("\nAnalysis completed:")
    print(f"Wells: {num_wells}")
    print(f"Reservoir zones: {total_zones}")
    print(f"Total thickness: {total_thickness:.1f} ft")

    return analysis_results


if __name__ == "__main__":
    """
    Entry point for well-log reservoir analysis pipeline.

    This script demonstrates how to run the full workflow:
    - Model prediction loading
    - Reservoir interpretation
    - Visualization generation
    - RMSE evaluation
    - Reservoir zone ranking
    """

    # =========================================================
    # Input configuration
    # =========================================================
    model_predictions_path = "model_predictions.csv"
    real_test_result_path = "real_test_results.csv"
    output_folder = "output"

    # =========================================================
    # Run full pipeline
    # =========================================================
    results = plot_well_predictions(
        model_predictions_path=model_predictions_path,
        real_test_result_path=real_test_result_path,
        output_folder=output_folder
    )

    # =========================================================
    # Print summary results
    # =========================================================
    print("\n" + "=" * 50)
    print("RESERVOIR ANALYSIS SUMMARY")
    print("=" * 50)

    for well_num, well_data in results["wells"].items():

        print(f"\nWell {well_num}")
        print("-" * 30)

        # Reservoir count
        zone_count = len(well_data["reservoir_zones"])
        print(f"Reservoir zones: {zone_count}")

        # RMSE
        avg_rmse = well_data["rmse_metrics"]["avg_rmse"]
        print(f"Average RMSE: {avg_rmse:.4f}")

        # Best reservoir zone
        if zone_count > 0:
            best_zone = max(
                well_data["reservoir_zones"],
                key=lambda x: x["avg_so"] * x["avg_phif"]
            )

            print(
                f"Best reservoir: R{best_zone['number']} "
                f"({best_zone['quality']})"
            )

    print("\nPipeline completed successfully.")