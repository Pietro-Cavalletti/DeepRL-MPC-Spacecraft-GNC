# =============================================================================
# Author: Pietro Cavalletti
# Year: 2025
# Description: Parses satellite docking statistics from text files and generates
#              comparative bar charts for different merit indices across orbits.
# =============================================================================

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.patches import Patch

def parse_stats_file(filepath):
    """
    Parses a text file containing docking statistics into a structured dictionary.

    Args:
        filepath (str): Path to the statistics text file.

    Returns:
        tuple: (parsed_data (dict), orbit_names (list))
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Specified file not found: {filepath}")

    # Mapping to maintain full names and units for plotting
    index_name_mapping = {
        "Consumo Energetico (Impulso Totale)": "Energy Consumption (Total Impulse) [m/s]",
        "Tempo di Calcolo Relativo": "Relative Computation Time [%]",
        "Tempo Medio di Calcolo per Step": "Mean Computation Time per Step [s]",
        "Violazione Vincolo Cono (Integrale)": "Cone Constraint Violation (Integral)"
    }
    
    parsed_data = {v: {'means': [], 'stds': []} for v in index_name_mapping.values()}
    orbit_names = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Detect new orbit section
            if "ORBITA" in line:
                match = re.search(r'\((.*?)\)', line)
                if match:
                    orbit_name = match.group(1)
                    if orbit_name not in orbit_names:
                        orbit_names.append(orbit_name)
            
            # Extract Mean and Std Dev data
            elif line.startswith('-'):
                try:
                    name_match = re.search(r'- (.*?):', line)
                    mean_match = re.search(r'Media = ([\d.e+-]+)', line)
                    std_match = re.search(r'Std Dev = ([\d.e+-]+)', line)
                    
                    if name_match and mean_match and std_match:
                        clean_name = name_match.group(1).strip()
                        full_name = index_name_mapping[clean_name]
                        
                        parsed_data[full_name]['means'].append(float(mean_match.group(1)))
                        parsed_data[full_name]['stds'].append(float(std_match.group(1)))

                except (KeyError, IndexError, ValueError) as e:
                    print(f"WARNING: Could not process line: '{line}'. Error: {e}")

    return parsed_data, orbit_names

def plot_from_parsed_data(parsed_data, orbit_names, output_dir=None):
    """
    Generates bar charts from parsed data.
    """
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "docking_results")

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    n_orbits = len(orbit_names)
    n_points_per_orbit = 4  # Standard: Theta = 0, 90, 180, 270
    
    # Color palettes for different indices
    color_palettes = {
        "Energy Consumption (Total Impulse) [m/s]": plt.get_cmap('Greens')(np.linspace(0.3, 0.9, n_points_per_orbit)),
        "Relative Computation Time [%]": plt.get_cmap('Blues')(np.linspace(0.3, 0.9, n_points_per_orbit)),
        "Mean Computation Time per Step [s]": plt.get_cmap('Purples')(np.linspace(0.3, 0.9, n_points_per_orbit)),
        "Cone Constraint Violation (Integral)": plt.get_cmap('Reds')(np.linspace(0.3, 0.9, n_points_per_orbit))
    }

    for english_name, data in parsed_data.items():
        if not data or not data.get('means'):
            continue

        bar_values = data['means']
        total_points = len(bar_values)
        current_colors = color_palettes[english_name]
        bar_colors = [current_colors[p % n_points_per_orbit] for p in range(total_points)]
        
        fig, ax = plt.subplots(figsize=(21, 9))
        x_positions = np.arange(total_points)
        
        ax.bar(x_positions, bar_values, color=bar_colors, alpha=0.85, edgecolor='black', width=0.8)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Group ticks and labels
        tick_positions = [i * n_points_per_orbit + (n_points_per_orbit - 1) / 2.0 for i in range(n_orbits)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(orbit_names, rotation=45, ha="right", fontsize=16)
        ax.tick_params(axis='y', labelsize=15)
        
        # Vertical separators between orbits
        for i in range(n_orbits - 1):
            separator_pos = (i + 1) * n_points_per_orbit - 0.5
            ax.axvline(x=separator_pos, color='grey', linestyle='--', linewidth=1.0, alpha=0.8)

        # Legend for Theta values
        legend_labels = [f"θ = {i*90}°" for i in range(n_points_per_orbit)]
        legend_patches = [Patch(color=current_colors[i], label=legend_labels[i]) for i in range(n_points_per_orbit)]
        ax.legend(handles=legend_patches, loc='upper right', fontsize=17)

        fig.tight_layout()
        
        # Safe filename creation
        safe_name = english_name.replace(" ", "_").replace("[", "").replace("]", "").replace("/", "").replace("%", "")
        fig.savefig(os.path.join(output_dir, f"barchart_{safe_name}_{timestamp}.png"))
        plt.close(fig)
            
    print(f"Charts successfully saved in: {output_dir}")

if __name__ == "__main__":
    stats_filename = "merit_indices_stats_detailed.txt"
    
    try:
        data, orbits = parse_stats_file(stats_filename)
        plot_from_parsed_data(data, orbits)
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")