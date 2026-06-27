import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib as mpl
import io
import sys
from contextlib import redirect_stdout


def setup_plotting_environment():
    """Configure plotting environment, including font settings and editable vector text settings."""

    mpl.rcParams.update({
        # Font settings
        "font.family": "Arial",

        # Keep TrueType fonts in PDF to avoid converting text to outlines
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # Keep text as editable text in SVG instead of converting it to paths
        "svg.fonttype": "none",

        # Prevent minus sign rendering issues
        "axes.unicode_minus": False,
    })

    # Create output directory
    os.makedirs("Results/Pictures", exist_ok=True)


def load_data(filename, na_value=-9999):
    """
    Load well log data and perform basic preprocessing.

    Parameters:
    - filename: input data file name
    - na_value: missing value marker

    Returns:
    - processed DataFrame
    """
    # Read data and handle missing values
    df = pd.read_csv(filename, na_values=na_value)
    # Drop only rows with missing DEPTH or WELLNUM; keep missing values in other features
    df.dropna(subset=['DEPTH', 'WELLNUM'], inplace=True)
    df['WELLNUM'] = df['WELLNUM'].astype(str).str.strip()
    return df


def is_plottable(series, min_data_percentage=0.1):
    """
    Check whether a feature has enough valid data for plotting.

    Parameters:
    - series: input data series
    - min_data_percentage: minimum ratio of non-missing values, default is 10%

    Returns:
    - bool indicating whether the feature is plottable
    """
    return series.notna().sum() / len(series) > min_data_percentage


def identify_reservoir_zones(df_well, gr_cutoff=60, gap_threshold=10, min_thickness=5, verbose=True):
    """
    Identify potential reservoir intervals using the GR cutoff method.

    Parameters:
    - df_well: well data indexed by DEPTH
    - gr_cutoff: GR threshold; values below this threshold are considered potential reservoirs
    - gap_threshold: maximum gap between consecutive points; larger gaps start a new reservoir interval
    - min_thickness: minimum reservoir thickness in feet; thinner intervals are filtered out
    - verbose: whether to print detailed information

    Returns:
    - list of reservoir intervals, each as a tuple (top_depth, bottom_depth)
    - reservoir identification report dictionary with detailed information
    """
    reservoir_report = {
        "identification_method": "Gamma Ray (GR) Cutoff Method",
        "parameters": {
            "gr_cutoff": gr_cutoff,
            "gap_threshold": gap_threshold,
            "min_thickness": min_thickness
        },
        "criteria": f"Reservoir is identified where GR < {gr_cutoff} API, with minimum thickness of {min_thickness}ft",
        "error": None,
        "raw_zones_count": 0,
        "filtered_zones_count": 0,
        "reservoir_zones": []
    }

    if 'GR' not in df_well.columns or not is_plottable(df_well['GR']):
        error_msg = "'GR' curve not found or insufficient data"
        reservoir_report["error"] = error_msg
        if verbose:
            print(f"⚠️ {error_msg}")
        return [], reservoir_report

    # Ensure data are sorted by depth
    df_gr = df_well['GR'].dropna().sort_index()

    if len(df_gr) == 0:
        error_msg = "No valid GR data after filtering"
        reservoir_report["error"] = error_msg
        if verbose:
            print(f"⚠️ {error_msg}")
        return [], reservoir_report

    if verbose:
        print(f"\n=== Reservoir Identification Criteria ===")
        print(f"Method: Gamma Ray (GR) Cutoff")
        print(f"GR cutoff: {gr_cutoff} API (values below this are potential reservoirs)")
        print(f"Minimum thickness: {min_thickness} ft")
        print(f"Gap threshold: {gap_threshold} ft (gaps larger than this split reservoirs)")
        print(f"Analyzed depth range: {df_gr.index.min():.1f} - {df_gr.index.max():.1f} ft")
        print(f"GR range in data: {df_gr.min():.1f} - {df_gr.max():.1f} API")

    # Find all points below the GR threshold
    reservoir_points = df_gr.index[df_gr < gr_cutoff].tolist()

    if not reservoir_points:
        error_msg = f"No points with GR < {gr_cutoff} API found"
        reservoir_report["error"] = error_msg
        if verbose:
            print(f"⚠️ {error_msg}")
        return [], reservoir_report

    # Group continuous points into reservoir intervals
    raw_reservoir_zones = []
    start_depth = reservoir_points[0]
    prev_depth = start_depth

    for depth in reservoir_points[1:]:
        # If the gap from the previous point exceeds the threshold, start a new reservoir interval
        if abs(depth - prev_depth) > gap_threshold:
            raw_reservoir_zones.append((start_depth, prev_depth))
            start_depth = depth
        prev_depth = depth

    # Add the last reservoir interval
    raw_reservoir_zones.append((start_depth, prev_depth))
    reservoir_report["raw_zones_count"] = len(raw_reservoir_zones)

    # Filter out reservoir intervals that are too thin
    filtered_reservoir_zones = [(top, bottom) for top, bottom in raw_reservoir_zones
                                if abs(bottom - top) >= min_thickness]

    # Sort reservoir intervals by depth
    filtered_reservoir_zones.sort(key=lambda x: x[0])
    reservoir_report["filtered_zones_count"] = len(filtered_reservoir_zones)

    # Generate detailed reservoir interval information
    zone_details = []
    for i, (top, bottom) in enumerate(filtered_reservoir_zones):
        thickness = bottom - top
        avg_gr = df_gr[(df_gr.index >= top) & (df_gr.index <= bottom)].mean()
        zone_info = {
            "zone_number": i + 1,
            "top_depth": round(top, 1),
            "bottom_depth": round(bottom, 1),
            "thickness": round(thickness, 1),
            "avg_gr": round(avg_gr, 1) if not pd.isna(avg_gr) else None
        }
        zone_details.append(zone_info)

    reservoir_report["reservoir_zones"] = zone_details

    if verbose:
        print(f"\n=== Reservoir Identification Results ===")
        print(f"Initially identified {len(raw_reservoir_zones)} potential zones")
        print(f"After thickness filtering, identified {len(filtered_reservoir_zones)} reservoir zones")

        if filtered_reservoir_zones:
            print("\nReservoir Zones:")
            print(f"{'Zone':<5}{'Top (ft)':<12}{'Bottom (ft)':<12}{'Thickness (ft)':<15}{'Avg GR (API)':<15}")
            print("-" * 59)

            for zone in zone_details:
                print(f"R{zone['zone_number']:<4}{zone['top_depth']:<12}{zone['bottom_depth']:<12}"
                      f"{zone['thickness']:<15}{zone['avg_gr']:<15}")

            total_thickness = sum(zone['thickness'] for zone in zone_details)
            print(f"\nTotal reservoir thickness: {total_thickness:.1f} ft")
            print(f"Percentage of reservoir in analyzed interval: "
                  f"{(total_thickness / (df_gr.index.max() - df_gr.index.min())) * 100:.1f}%")

    return filtered_reservoir_zones, reservoir_report


def create_track1_gr_cali(ax, df_well, gr_color='green', cali_color='blue', gr_xlim=(0, 150), cali_xlim=(8, 10)):
    """
    Create track 1 curves: GR and CALI.

    Parameters:
    - ax: plotting axis object
    - df_well: well data
    - gr_color: color of the GR curve
    - cali_color: color of the CALI curve
    - gr_xlim: x-axis range tuple for GR (min, max)
    - cali_xlim: x-axis range tuple for CALI (min, max)
    """
    # Create primary axis for GR
    if is_plottable(df_well['GR']):
        valid_data = df_well[['GR']].dropna()
        ax.plot(valid_data['GR'], valid_data.index, color=gr_color, label='GR (API)')

    # Create secondary x-axis for CALI
    ax_cali = ax.twiny()

    # Configure CALI axis with ticks on top
    ax_cali.xaxis.set_label_position('top')
    ax_cali.xaxis.set_ticks_position('top')
    ax_cali.set_xlabel('CALI (in)')

    # Set GR axis range and ticks
    gr_min, gr_max = gr_xlim
    ax.set_xlim(gr_min, gr_max)

    # Ensure start and end values are shown as ticks
    num_ticks = 6  # Number of ticks including start and end points
    gr_ticks = np.linspace(gr_min, gr_max, num_ticks)
    ax.set_xticks(gr_ticks)
    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))

    ax.set_xlabel("GR (API)")

    # Plot CALI curve and set its range
    if is_plottable(df_well['CALI']):
        valid_data = df_well[['CALI']].dropna()
        ax_cali.plot(valid_data['CALI'], valid_data.index, color=cali_color, label='CALI (in)')

    # Set CALI axis range and ticks
    cali_min, cali_max = cali_xlim
    ax_cali.set_xlim(cali_min, cali_max)

    # Ensure start and end values are shown as ticks
    num_ticks = 5  # Number of ticks including start and end points
    cali_ticks = np.linspace(cali_min, cali_max, num_ticks)
    ax_cali.set_xticks(cali_ticks)
    ax_cali.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))

    # Add legend in the upper-right corner with vertical layout
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax_cali.get_legend_handles_labels()
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2

    # Add legend if legend items exist
    if all_handles:
        ax.legend(all_handles, all_labels, loc='upper right', bbox_to_anchor=(0.98, 0.98),
                  framealpha=0.8, fontsize=8, ncol=1)  # ncol=1 means vertical layout

    ax.grid(True)

    return ax, ax_cali


def create_track2_resistivity(ax, df_well, rdep_color='red', rmed_color='black'):
    """
    Create track 2 resistivity curves.

    Parameters:
    - ax: plotting axis object
    - df_well: well data
    - rdep_color: color of the RDEP curve
    - rmed_color: color of the RMED curve
    """
    if is_plottable(df_well['RDEP']):
        valid_data = df_well[['RDEP']].dropna()
        ax.semilogx(valid_data['RDEP'], valid_data.index, color=rdep_color, label='RDEP')
    if is_plottable(df_well['RMED']):
        valid_data = df_well[['RMED']].dropna()
        ax.semilogx(valid_data['RMED'], valid_data.index, color=rmed_color, linestyle='--', label='RMED')

    # Set ticks for the logarithmic axis
    # Set explicit logarithmic tick points for resistivity to include 0.2 and 200
    ax.set_xlim(0.2, 200)

    # Use logarithmic tick points and ensure the start and end values 0.2 and 200 are included
    log_ticks = [0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200]
    ax.set_xticks(log_ticks)

    # Use ScalarFormatter instead of LogFormatter to display actual values
    formatter = mpl.ticker.ScalarFormatter()
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)

    # Place major ticks at the top
    ax.xaxis.set_tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Place axis label at the top
    ax.xaxis.set_label_position('top')
    ax.set_xlabel("Resistivity (Ohm·m)")

    # Add legend in the upper-right corner with vertical layout
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98),
                  framealpha=0.8, fontsize=8, ncol=1)  # ncol=1 means vertical layout

    ax.grid(True, which='both', linestyle='-', linewidth=0.5)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)


def create_track3_porosity(ax, df_well, dtc_color='blue', neu_color='purple', den_color='brown',
                           dtc_xlim=(200, 0), neu_xlim=(0.45, -0.15), den_xlim=(1.85, 2.85)):
    """
    Create track 3 porosity-related curves (DTC/NEU/DEN).

    Parameters:
    - ax: main plotting axis object
    - df_well: well data
    - dtc_color: color of the DTC curve
    - neu_color: color of the NEU curve
    - den_color: color of the DEN curve
    - dtc_xlim: x-axis range tuple for DTC (min, max)
    - neu_xlim: x-axis range tuple for NEU (min, max)
    - den_xlim: x-axis range tuple for DEN (min, max)

    Returns:
    - created three axis objects
    """
    # Keep the original axis object without deleting it
    ax_orig = ax

    # Create three overlaid axes
    ax_dtc = ax_orig  # Bottom axis for DTC
    ax_neu = ax_orig.twiny()  # Create the second x-axis for NEU
    ax_den = ax_orig.twiny()  # Create the third x-axis for DEN

    # Configure DTC axis with bottom ticks
    ax_dtc.xaxis.set_label_position('bottom')
    ax_dtc.xaxis.set_ticks_position('bottom')
    ax_dtc.set_xlabel('DTC')

    # Set DTC axis range and ticks
    dtc_min, dtc_max = dtc_xlim  # DTC axis is reversed (200 to 0)
    ax_dtc.set_xlim(dtc_min, dtc_max)

    # Ensure start and end values are shown as ticks
    num_ticks = 6  # Number of ticks including start and end points
    dtc_ticks = np.linspace(dtc_min, dtc_max, num_ticks)
    ax_dtc.set_xticks(dtc_ticks)
    ax_dtc.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))

    # Configure NEU axis with an additional bottom tick row
    ax_neu.xaxis.set_label_position('bottom')
    ax_neu.xaxis.set_ticks_position('bottom')
    ax_neu.spines['bottom'].set_position(('outward', 40))  # Move downward by 40 points
    ax_neu.set_xlabel('NEU')

    # Set NEU axis range and ticks
    neu_min, neu_max = neu_xlim  # NEU axis is also reversed (0.45 to -0.15)
    ax_neu.set_xlim(neu_min, neu_max)

    # Ensure start and end values are shown as ticks
    num_ticks = 5  # Number of ticks including start and end points
    neu_ticks = np.linspace(neu_min, neu_max, num_ticks)
    ax_neu.set_xticks(neu_ticks)
    ax_neu.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))

    # Configure DEN axis with top ticks
    ax_den.xaxis.set_label_position('top')
    ax_den.xaxis.set_ticks_position('top')
    ax_den.set_xlabel('DEN')

    # Set DEN axis range and ticks
    den_min, den_max = den_xlim
    ax_den.set_xlim(den_min, den_max)

    # Ensure start and end values are shown as ticks
    num_ticks = 6  # Number of ticks including start and end points
    den_ticks = np.linspace(den_min, den_max, num_ticks)
    ax_den.set_xticks(den_ticks)
    ax_den.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))

    # Plot DTC curve
    if is_plottable(df_well['DTC']):
        valid_data = df_well[['DTC']].dropna()
        ax_dtc.plot(valid_data['DTC'], valid_data.index, color=dtc_color, label='DTC')

    # Plot NEU curve
    if is_plottable(df_well['NEU']):
        valid_data = df_well[['NEU']].dropna()
        ax_neu.plot(valid_data['NEU'], valid_data.index, color=neu_color, label='NEU')

    # Plot DEN curve
    if is_plottable(df_well['DEN']):
        valid_data = df_well[['DEN']].dropna()
        ax_den.plot(valid_data['DEN'], valid_data.index, color=den_color, label='DEN')

    # Collect all legend items
    handles1, labels1 = ax_dtc.get_legend_handles_labels()
    handles2, labels2 = ax_neu.get_legend_handles_labels()
    handles3, labels3 = ax_den.get_legend_handles_labels()

    all_handles = handles1 + handles2 + handles3
    all_labels = labels1 + labels2 + labels3

    # Add a unified legend if legend items exist
    if all_handles:
        ax_dtc.legend(all_handles, all_labels, loc='upper right', bbox_to_anchor=(0.98, 0.98),
                      framealpha=0.8, fontsize=8, ncol=1)  # ncol=1 means vertical layout

    # Add range information box
    box_props = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.7)
    ax_dtc.text(0.98, 0.02,
                f"DTC: {dtc_xlim[0]}-{dtc_xlim[1]}\n"
                f"NEU: {neu_xlim[0]}-{neu_xlim[1]}\n"
                f"DEN: {den_xlim[0]}-{den_xlim[1]}",
                transform=ax_dtc.transAxes,
                horizontalalignment='right',
                verticalalignment='bottom',
                fontsize=7,
                bbox=box_props)

    return ax_dtc, ax_neu, ax_den


def mark_reservoir_zones(axs, reservoir_zones, label_fontsize=7):
    """
    Mark reservoir intervals on the plot.

    Parameters:
    - axs: list of plotting axis objects
    - reservoir_zones: list of reservoir intervals
    - label_fontsize: label font size
    """
    for i, (top, bottom) in enumerate(reservoir_zones):
        for ax in axs:
            height = bottom - top
            # Mark reservoir intervals on each track using semi-transparent yellow rectangles
            rect = Rectangle(
                (ax.get_xlim()[0], top),  # Upper-left coordinate
                ax.get_xlim()[1] - ax.get_xlim()[0],  # Width spans the full x-axis range
                height,  # Height is the reservoir interval thickness
                facecolor='yellow',
                alpha=0.2,
                zorder=0  # Ensure the rectangle is below the curves
            )
            ax.add_patch(rect)

            # Annotate reservoir number and thickness on the first track only
            if ax == axs[0]:
                mid_depth = top + height / 2
                ax.text(
                    ax.get_xlim()[1] * 0.8,  # x-position at 80% of the track width
                    mid_depth,  # y-position at the middle of the reservoir interval
                    f"R{i + 1}\n{height:.0f}ft",  # Remove decimals to save space
                    fontsize=label_fontsize,
                    ha='right',  # Right aligned
                    va='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)  # Reduce padding
                )


def add_reservoir_info_table(fig, well, reservoir_zones, split_threshold=6, fontsize=8):
    """
    Add reservoir information table at the bottom of the figure.

    Parameters:
    - fig: figure object
    - well: well number
    - reservoir_zones: list of reservoir intervals
    - split_threshold: threshold number of reservoirs for two-column display
    - fontsize: font size
    """
    # Add reservoir information to the title
    if reservoir_zones:
        zone_text = f" • {len(reservoir_zones)} Reservoir Zones Identified"
        fig.suptitle(f"Well Log Plot - WELLNUM: {well}{zone_text}", fontsize=18, y=0.98)  # Set y to 0.98 to move the title upward

        # Add reservoir interval table with improved layout to avoid text overlap
        # Split the list into two columns
        zones_count = len(reservoir_zones)
        if zones_count > split_threshold:  # If the number of reservoirs exceeds the threshold, use two-column display
            col1 = reservoir_zones[:zones_count // 2]
            col2 = reservoir_zones[zones_count // 2:]

            # First column
            zone_info_col1 = "\n".join([f"Reservoir R{i + 1}: {top:.1f}-{bottom:.1f}ft ({bottom - top:.1f}ft)"
                                        for i, (top, bottom) in enumerate(col1)])
            # Second column
            zone_info_col2 = "\n".join([f"R{i + 1 + len(col1)}: {top:.1f}-{bottom:.1f}ft ({bottom - top:.1f}ft)"
                                        for i, (top, bottom) in enumerate(col2)])

            # Place the first column at the lower-left using a larger y value
            fig.text(0.02, 0.03, zone_info_col1, fontsize=fontsize, verticalalignment='bottom')
            # Place the second column to the right of the first column
            fig.text(0.25, 0.03, zone_info_col2, fontsize=fontsize, verticalalignment='bottom')
        else:
            # Use single-column display if the number of reservoirs is below the threshold
            zone_info = "\n".join([f"Reservoir R{i + 1}: {top:.1f}-{bottom:.1f}ft ({bottom - top:.1f}ft)"
                                   for i, (top, bottom) in enumerate(reservoir_zones)])
            fig.text(0.02, 0.03, zone_info, fontsize=fontsize, verticalalignment='bottom')
    else:
        # If no reservoir intervals are identified
        fig.suptitle(f"Well Log Plot - WELLNUM: {well}", fontsize=18, y=0.98)  # Set y to 0.98 to move the title upward


def setup_depth_ticks(axs, min_depth, max_depth):
    """
    Configure depth ticks and grid lines with dynamic tick interval adjustment.

    Parameters:
    - axs: list of plotting axis objects
    - min_depth: minimum depth
    - max_depth: maximum depth

    Returns:
    - tuple: (major_ticks, minor_ticks) major and minor tick arrays
    """
    depth_span = max_depth - min_depth

    # Calculate an appropriate depth interval
    if depth_span > 2000:
        depth_interval = 200
    elif depth_span > 1000:
        depth_interval = 100
    else:
        depth_interval = 50

    # Generate major depth ticks
    major_ticks = np.arange(
        np.floor(min_depth / depth_interval) * depth_interval,
        np.ceil(max_depth / depth_interval) * depth_interval + 1,
        depth_interval
    )

    # Generate minor depth ticks
    minor_interval = depth_interval / 5  # Divide each major interval into five minor intervals
    minor_ticks = np.arange(
        np.floor(min_depth / minor_interval) * minor_interval,
        np.ceil(max_depth / minor_interval) * minor_interval + 1,
        minor_interval
    )

    # Set common properties for all axes
    for i, ax in enumerate(axs):
        # Set y-axis range so depth increases from top to bottom
        ax.set_ylim(max_depth, min_depth)

        # Set major and minor depth ticks
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)

        # Ensure grid lines are displayed correctly
        ax.grid(True, which='major', axis='both')
        ax.grid(True, which='minor', axis='y', alpha=0.5)

        # Show depth labels only on the first and last axes
        if i == 0:  # First axis
            ax.set_ylabel('Depth (ft)', fontsize=12, fontweight='bold')
            # Ensure depth labels are displayed on the first axis
            ax.set_yticklabels([f"{int(tick)}" for tick in major_ticks])
        elif i == len(axs) - 1:  # Last axis
            # Display depth labels on the right side of the last axis
            ax_twin = ax.twinx()
            ax_twin.set_ylim(max_depth, min_depth)
            ax_twin.set_yticks(major_ticks)
            ax_twin.set_yticklabels([f"{int(tick)}" for tick in major_ticks])
            ax_twin.set_ylabel('')  # Do not show an additional y-axis label
        else:
            ax.set_yticklabels([])  # Do not show depth labels on middle axes

    return major_ticks, minor_ticks


def add_horizontal_depth_scale(fig, depth_scale_length):
    """
    Add a horizontal depth scale above the figure.

    Parameters:
    - fig: figure object
    - depth_scale_length: depth length represented by the scale in feet
    """
    # Add text-based scale above the figure
    scale_text = f"Depth Scale: {depth_scale_length} ft"

    # Add scale label below the title and close to the top
    fig.text(0.5, 0.95, scale_text,
             ha='center', va='top', fontsize=10,
             bbox=dict(facecolor='white', edgecolor='black', alpha=0.7, pad=5))


def add_depth_markers(axs, depths, labels=None, colors=None, linewidth=1, alpha=0.5, fontsize=8):
    """
    Add depth marker lines and labels to the plot.

    Parameters:
    - axs: list of plotting axis objects
    - depths: list of depths to be marked
    - labels: corresponding labels; if None, depth values are used
    - colors: corresponding colors; if None, default colors are used
    - linewidth: line width
    - alpha: transparency
    - fontsize: font size
    """
    if labels is None:
        labels = [f"{depth:.1f}ft" for depth in depths]

    if colors is None:
        colors = ['black'] * len(depths)

    for depth, label, color in zip(depths, labels, colors):
        for i, ax in enumerate(axs):
            # Draw horizontal line
            ax.axhline(y=depth, color=color, linestyle='--', linewidth=linewidth, alpha=alpha)

            # Add text labels only on the first track
            if i == 0:
                ax.text(ax.get_xlim()[0] * 1.01, depth, label,
                        ha='left', va='center', fontsize=fontsize,
                        color=color, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))


def plot_well_log(df, filename="test.csv", output_dir="Results/Pictures",
                  identify_reservoirs=True,
                  gr_cutoff=60, min_thickness=5, gap_threshold=10,
                  na_value=-9999,
                  figsize=None,  # Set dynamically
                  dpi=300, verbose=True,
                  gr_xlim=(0, 150), cali_xlim=(8, 10), dtc_xlim=(200, 0),
                  neu_xlim=(0.45, -0.15), den_xlim=(1.85, 2.85),
                  mark_intervals=True,  # Whether to add depth interval markers
                  interval_value=200  # Depth interval value in feet
                  ):
    """
    Main function for plotting well log curves with optional reservoir interval marking.

    Parameters:
    - df: processed DataFrame or data file name
    - filename: data file name if df is not provided
    - output_dir: output directory
    - identify_reservoirs: whether to automatically identify and mark reservoir intervals
    - gr_cutoff: GR threshold; values below this threshold are considered potential reservoirs
    - min_thickness: minimum reservoir thickness in feet
    - gap_threshold: maximum gap between consecutive points
    - na_value: missing value marker
    - figsize: figure size; if None, it is dynamically set based on depth range
    - dpi: figure resolution
    - verbose: whether to print detailed information
    - gr_xlim: x-axis range tuple for GR (min, max)
    - cali_xlim: x-axis range tuple for CALI (min, max)
    - dtc_xlim: x-axis range tuple for DTC (min, max)
    - neu_xlim: x-axis range tuple for NEU (min, max)
    - den_xlim: x-axis range tuple for DEN (min, max)
    - mark_intervals: whether to add depth interval markers
    - interval_value: depth interval value in feet

    Returns:
    - dictionary containing reservoir information for each well
    """
    # Set up plotting environment
    setup_plotting_environment()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load data if needed
    if isinstance(df, str):
        if verbose:
            print(f"Loading data from file: {df}")
        df = load_data(df, na_value=na_value)

    # Result report dictionary
    well_reports = {}

    if verbose:
        print(f"\n{'=' * 30}")
        print(f"WELL LOG ANALYSIS{' - RESERVOIR IDENTIFICATION' if identify_reservoirs else ''}")
        print(f"{'=' * 30}")
        if identify_reservoirs:
            print(f"Analysis parameters:")
            print(f"  - GR cutoff: {gr_cutoff} API")
            print(f"  - Minimum thickness: {min_thickness} ft")
            print(f"  - Gap threshold: {gap_threshold} ft")
        print(f"Curve display ranges:")
        print(f"  - GR: {gr_xlim[0]}-{gr_xlim[1]} API")
        print(f"  - CALI: {cali_xlim[0]}-{cali_xlim[1]} in")
        print(f"  - DTC: {dtc_xlim[0]}-{dtc_xlim[1]}")
        print(f"  - NEU: {neu_xlim[0]}-{neu_xlim[1]}")
        print(f"  - DEN: {den_xlim[0]}-{den_xlim[1]}")
        print(f"{'=' * 30}\n")

    # Iterate through each well
    for well, df_well in df.groupby("WELLNUM"):
        if verbose:
            print(f"\n{'=' * 50}")
            print(f"Processing Well: {well}")
            print(f"{'=' * 50}")
            print(f"Depth range: {df_well['DEPTH'].min():.1f} - {df_well['DEPTH'].max():.1f} ft")
            print(f"Data points: {len(df_well):,}")

        df_well = df_well.sort_values(by="DEPTH").copy()
        df_well.set_index("DEPTH", inplace=True)

        # Get depth range
        y_min = df_well.index.min()
        y_max = df_well.index.max()
        depth_span = y_max - y_min

        # Dynamically calculate figure size based on plot_well_predictions.py
        if figsize is None:
            # Dynamically set figure height based on depth range
            if depth_span > 2000:
                fig_height = 24  # Use a taller figure for wells with a larger depth range
            elif depth_span > 1000:
                fig_height = 18
            else:
                fig_height = 12

            # Use fixed width, with height adjusted according to depth span
            fig_width = 15

            figsize = (fig_width, fig_height)

            if verbose:
                print(f"Dynamic figure size: {figsize}")

        # Reservoir interval identification if enabled
        reservoir_zones = []
        reservoir_report = {}
        if identify_reservoirs:
            reservoir_zones, reservoir_report = identify_reservoir_zones(
                df_well,
                gr_cutoff=gr_cutoff,
                gap_threshold=gap_threshold,
                min_thickness=min_thickness,
                verbose=verbose
            )

            # Add well number to report
            reservoir_report["well_number"] = well
            well_reports[well] = reservoir_report

        # Create figure and axes
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figsize, sharey=True)

        # Use the modified depth tick setup function
        major_ticks, minor_ticks = setup_depth_ticks(axs, y_min, y_max)

        # Create three tracks using custom ranges
        # Use the updated first-track function (GR + CALI)
        ax_gr, ax_cali = create_track1_gr_cali(axs[0], df_well, gr_xlim=gr_xlim, cali_xlim=cali_xlim)
        create_track2_resistivity(axs[1], df_well)
        dtc_ax, neu_ax, den_ax = create_track3_porosity(
            axs[2],
            df_well,
            dtc_xlim=dtc_xlim,
            neu_xlim=neu_xlim,
            den_xlim=den_xlim
        )

        # Set y-axis label
        axs[0].set_ylabel("Depth (ft)", fontsize=10, fontweight='bold')

        # Add depth interval markers if enabled
        if mark_intervals:
            # Calculate interval depth points, for example one marker every 200 ft
            start_depth = np.ceil(y_min / interval_value) * interval_value
            interval_depths = np.arange(start_depth, y_max, interval_value)
            interval_labels = [f"{int(depth)}ft" for depth in interval_depths]
            interval_colors = ['darkblue'] * len(interval_depths)

            # Add depth marker lines and labels
            add_depth_markers(axs, interval_depths, interval_labels, interval_colors,
                              linewidth=1.2, alpha=0.4, fontsize=9)

        # Mark reservoir intervals if enabled and identified
        if identify_reservoirs and reservoir_zones:
            mark_reservoir_zones(axs, reservoir_zones)
            # Add reservoir information table
            add_reservoir_info_table(fig, well, reservoir_zones)
        else:
            # Show well number only without reservoir information
            fig.suptitle(f"Well Log Plot - WELLNUM: {well}", fontsize=18, y=0.98)  # Set y to 0.98 to move the title upward

        # Add depth range information
        fig.text(0.5, 0.01, f"Depth range: {int(y_min)}-{int(y_max)} ft ({int(depth_span)} ft)",
                 ha='center', fontsize=9, style='italic')

        # Adjust layout and save outputs
        plt.tight_layout(rect=[0, 0.12, 1, 0.97])  # Set top margin to 0.97 and bottom margin to 0.12 to leave space for title and bottom reservoir information

        # Add legend and information labels
        fig.text(0.98, 0.01, f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d')}",
                 ha='right', fontsize=7, style='italic')

        plot_path = f"{output_dir}/Well_{well}.png"
        pdf_path = f"{output_dir}/Well_{well}.pdf"

        plt.savefig(plot_path, dpi=dpi, bbox_inches='tight')  # Save PNG
        plt.savefig(pdf_path, bbox_inches='tight')  # Save PDF; vector format is suitable for papers and PPT editing
        plt.close()

        if verbose:
            print(f"\nWell log plot saved to: {plot_path}")
            print(f"Well log PDF saved to: {pdf_path}")

    # Summarize results
    if identify_reservoirs:
        total_zones = sum(len(report["reservoir_zones"]) for report in well_reports.values())
        total_thickness = sum(
            sum(zone["thickness"] for zone in report["reservoir_zones"])
            for report in well_reports.values()
        )

        if verbose and well_reports:
            print("\n" + "=" * 50)
            print("SUMMARY OF RESERVOIR IDENTIFICATION")
            print("=" * 50)
            print(f"Wells analyzed: {len(well_reports)}")
            print(f"Total reservoir zones identified: {total_zones}")
            print(f"Total reservoir thickness: {total_thickness:.1f} ft")
            print(f"Average zones per well: {total_zones / len(well_reports) if well_reports else 0:.1f}")
            print(f"Average thickness per zone: {total_thickness / total_zones if total_zones else 0:.1f} ft")
            print("=" * 50)
            print("\n✅ All well log plots completed with three tracks: GR+CALI, Resistivity, and combined DTC/NEU/DEN.")
            if identify_reservoirs:
                print(
                    f"✅ Reservoir zones identified (GR < {gr_cutoff} API, min thickness: {min_thickness}ft) and marked with yellow transparent regions.")
            print(f"✅ Curve ranges set to: GR={gr_xlim}, CALI={cali_xlim}, DTC={dtc_xlim}, NEU={neu_xlim}, DEN={den_xlim}")
            print(f"✅ Depth visualization enhanced with dynamic scale based on depth range")
            if mark_intervals:
                print(f"✅ Depth interval markers added every {interval_value}ft")

    return well_reports


def save_reservoir_report(well_reports, output_file="Results/reservoir_analysis_report.txt"):
    """
    Save reservoir analysis results to a text file.

    Parameters:
    - well_reports: well report dictionary
    - output_file: output file name

    Returns:
    - output file path
    """
    # Return if there are no reservoir reports
    if not well_reports:
        print("No reservoir reports to save.")
        return None

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("==========================================\n")
        f.write("      WELL LOG RESERVOIR ANALYSIS REPORT       \n")
        f.write("==========================================\n\n")

        f.write("ANALYSIS PARAMETERS:\n")
        if well_reports:
            sample_report = next(iter(well_reports.values()))
            f.write(f"Method: {sample_report['identification_method']}\n")
            f.write(f"GR cutoff: {sample_report['parameters']['gr_cutoff']} API\n")
            f.write(f"Minimum thickness: {sample_report['parameters']['min_thickness']} ft\n")
            f.write(f"Gap threshold: {sample_report['parameters']['gap_threshold']} ft\n\n")

        # Summary statistics
        total_zones = sum(len(report["reservoir_zones"]) for report in well_reports.values())
        total_thickness = sum(
            sum(zone["thickness"] for zone in report["reservoir_zones"])
            for report in well_reports.values()
        )

        f.write("SUMMARY STATISTICS:\n")
        f.write(f"Wells analyzed: {len(well_reports)}\n")
        f.write(f"Total reservoir zones identified: {total_zones}\n")
        f.write(f"Total reservoir thickness: {total_thickness:.1f} ft\n")
        f.write(f"Average zones per well: {total_zones / len(well_reports) if well_reports else 0:.1f}\n")
        f.write(f"Average thickness per zone: {total_thickness / total_zones if total_zones else 0:.1f} ft\n\n")

        # Detailed information for each well
        for well, report in well_reports.items():
            f.write(f"\n{'=' * 50}\n")
            f.write(f"WELL: {well}\n")
            f.write(f"{'=' * 50}\n")

            if report["error"]:
                f.write(f"ERROR: {report['error']}\n")
                continue

            f.write(f"Initially identified zones: {report['raw_zones_count']}\n")
            f.write(f"Zones after thickness filtering: {report['filtered_zones_count']}\n\n")

            if report["reservoir_zones"]:
                f.write(f"{'Zone':<5}{'Top (ft)':<12}{'Bottom (ft)':<12}{'Thickness (ft)':<15}{'Avg GR (API)':<15}\n")
                f.write(f"{'-' * 59}\n")

                for zone in report["reservoir_zones"]:
                    f.write(f"R{zone['zone_number']:<4}{zone['top_depth']:<12}{zone['bottom_depth']:<12}"
                            f"{zone['thickness']:<15}{zone['avg_gr']:<15}\n")

                total_well_thickness = sum(zone["thickness"] for zone in report["reservoir_zones"])
                f.write(f"\nTotal reservoir thickness for well {well}: {total_well_thickness:.1f} ft\n")
            else:
                f.write("No reservoir zones identified for this well.\n")

    print(f"✅ Detailed reservoir analysis report saved to: {output_file}")
    return output_file


def plot_well_log_data(data_file="test.csv",
                       output_dir="Results/Pictures",
                       identify_reservoirs=True,
                       gr_cutoff=60,
                       min_thickness=5,
                       gap_threshold=10,
                       verbose=True,
                       gr_xlim=(0, 150),
                       cali_xlim=(8, 10),  # Update CALI range parameter to 8-10
                       dtc_xlim=(200, 0),
                       neu_xlim=(0.45, -0.15),
                       den_xlim=(1.85, 2.85),
                       figsize=None,
                       mark_intervals=True,
                       interval_value=200,
                       save_report=True):
    """
    Wrapper function for running well log analysis and plotting.

    Parameters:
    - data_file: CSV data file path
    - output_dir: directory for saving output figures
    - identify_reservoirs: whether to identify reservoir intervals
    - gr_cutoff: GR threshold for reservoir identification
    - min_thickness: minimum reservoir thickness
    - gap_threshold: gap threshold for reservoir segmentation
    - verbose: whether to print detailed information
    - gr_xlim: x-axis range for GR curve
    - cali_xlim: x-axis range for CALI curve
    - dtc_xlim: x-axis range for DTC curve
    - neu_xlim: x-axis range for NEU curve
    - den_xlim: x-axis range for DEN curve
    - figsize: figure size; dynamically calculated if None
    - mark_intervals: whether to mark depth intervals
    - interval_value: depth interval value
    - save_report: whether to save reservoir report

    Returns:
    - tuple: (well_reports, output_text)
        - well_reports: dictionary containing reservoir analysis results
        - output_text: string containing all printed output
    """
    # Redirect stdout to capture all print statements
    f = io.StringIO()
    with redirect_stdout(f):
        # Output directory
        os.makedirs(output_dir, exist_ok=True)

        if verbose:
            print("Starting to run the code!")

        # Load data
        if verbose:
            print(f"Loading data from file: {data_file}...")
        test_df = load_data(data_file)

        # Run analysis with options
        if verbose:
            if identify_reservoirs:
                print("Running well log analysis with reservoir identification...")
            else:
                print("Running well log analysis without reservoir identification...")

        well_reports = plot_well_log(
            test_df,
            output_dir=output_dir,
            identify_reservoirs=identify_reservoirs,
            gr_cutoff=gr_cutoff,
            min_thickness=min_thickness,
            gap_threshold=gap_threshold,
            verbose=verbose,
            gr_xlim=gr_xlim,
            cali_xlim=cali_xlim,
            dtc_xlim=dtc_xlim,
            neu_xlim=neu_xlim,
            den_xlim=den_xlim,
            figsize=figsize,
            mark_intervals=mark_intervals,
            interval_value=interval_value
        )

        # Save report if needed
        report_path = None
        if identify_reservoirs and well_reports and save_report:
            report_path = save_reservoir_report(well_reports)

        if verbose:
            print("Code run complete!")
            if report_path:
                print(f"Report saved at: {report_path}")

    # Get all captured output
    output_text = f.getvalue()

    # Return well reports and captured output
    return well_reports, output_text


# Example: how to use the function
if __name__ == "__main__":
    # Run with default parameters
    well_reports, output_text = plot_well_log_data()
    # Print captured output
    print("Execution log:")
    print("-" * 50)
    print(output_text)
    print("-" * 50)
    print("Execution completed!")