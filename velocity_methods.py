
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from cycler import cycler
from multi_run_stats import MultiRunStatsCreator

# Turn off warnings
warnings.filterwarnings("ignore")

# Define the properties of the plots
line_cycler   = (cycler(color=["#E69F00", "#8B008B", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                 cycler(linestyle=["-", "--", "-.", ":", "-", "--", "-."]))
marker_cycler = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                 cycler(linestyle=["none", "none", "none", "none", "none", "none", "none"]) +
                 cycler(marker=["4", "2", "3", "1", "+", "x", "."]))

# Set the properties of the plots and start subplots
plt.rc('axes', prop_cycle=line_cycler)


# Define core parameters for spike detection
num_runs = 100   # Number of runs for each threshold
sd_thresholds = np.arange(1.0, 7.01 , 0.05)  # Standard deviation thresholds for peak detection
signal_speeds = [1, 5, 10, 30, 60, 120] # m/s
loop_methods = ['AHC', 'SINC', 'SPLINE', 'CORR']

# Activate code to run the simulation (long runtime)
run_simulation = False

if run_simulation:
    # Run the simulation
    multi_run_stats = MultiRunStatsCreator(num_runs, sd_thresholds, signal_speeds)

    # Export to Excel
    multi_run_stats.statistics_df.to_excel('multi_run_stats.xlsx', index=False)
    multi_run_stats.multiple_run_signal_df.to_excel('multi_run_peaks.xlsx', index=False)


# Activate code to load the simulation data
load_simulation_data = True

# Load the simulation data
if load_simulation_data:
    multi_run_stats_df = pd.read_excel('multi_run_stats.xlsx')
    multiple_run_signal_df = pd.read_excel('multi_run_peaks.xlsx')

    # Change all values "nan" to np.nan
    multi_run_stats_df = multi_run_stats_df.replace("nan", np.nan)
    multiple_run_signal_df = multiple_run_signal_df.replace("nan", np.nan)


# Plot line charts for each velocity with different lines for methods and threshold on x-axis
for method in loop_methods:
    # Plot the data across all signal speeds
    total_df = multi_run_stats_df[(multi_run_stats_df['Velocity'] == "Total") &
                                  (multi_run_stats_df['Method'] == method)]

    plt.figure(figsize=(20, 13))
    plt.plot(total_df['SD Threshold'], total_df['A: True Spike, True Velocity'], label='A: True Spike,\nTrue Velocity')
    plt.plot(total_df['SD Threshold'], total_df['B: True Spike, Wrong Velocity'],
             label='B: True Spike,\nFalse Velocity')
    plt.plot(total_df['SD Threshold'], total_df['C: True Spike, Wrong Velocity (Infinite Velocity)'],
             label='C: True Spike,\nFalse Velocity\n(Inf. Velocity)')
    plt.plot(total_df['SD Threshold'], total_df['D: Missed Spike'], label='D: Missed\nSpike')
    plt.plot(total_df['SD Threshold'], total_df['E: False Spike'], label='E: False Spike')
    plt.xlim(2, 7)
    plt.ylim(0, 1)
    plt.xlabel('σ Threshold')
    plt.ylabel('Rate')
    plt.title(f'All Velocities ({method})')
    plt.legend()
    plt.show()

    # Create a 2x3 grid of subplots
    fig, axs = plt.subplots(2, 3, figsize=(20, 13))  # Create a 2x3 grid of subplots
    axs = axs.flatten()
    plot_index = 0

    for signal_speed in signal_speeds:
        if plot_index >= len(axs):
            break  # Stop if there are no more subplots available

        # Filter the signal dataframe for the current signal speed and method
        signal_df = multi_run_stats_df[(multi_run_stats_df['Velocity'] == signal_speed) &
                                       (multi_run_stats_df['Method'] == method)]

        # Select the appropriate subplot
        ax = axs[plot_index]

        # Plot the data
        ax.plot(signal_df['SD Threshold'], signal_df['A: True Spike, True Velocity'],
                label='A: True Spike,\nTrue Velocity')
        ax.plot(signal_df['SD Threshold'], signal_df['B: True Spike, Wrong Velocity'],
                label='B: True Spike,\nFalse Velocity')
        ax.plot(signal_df['SD Threshold'], signal_df['C: True Spike, Wrong Velocity (Infinite Velocity)'],
                label='C: True Spike,\nFalse Velocity\n(Inf. Velocity)')
        ax.plot(signal_df['SD Threshold'], signal_df['D: Missed Spike'], label='D: Missed\nSpike')
        ax.set_xlim(2, 7)
        ax.set_ylim(0, 1)
        ax.set_xlabel('σ Threshold')
        ax.set_ylabel('Rate')

        # Set the title inside the plot using ax.text()
        ax.text(0.5, 0.95, f'{signal_speed} m/s',
                horizontalalignment='center',
                verticalalignment='top',
                transform=ax.transAxes,
                fontsize=12,
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        # Only show the y-axis for graphs 1 and 4
        if plot_index not in [0, 3]:
            ax.yaxis.set_visible(False)

        # Only show the x-axis for graphs 4-6
        if plot_index < 3:
            ax.xaxis.set_visible(False)

        # Only show the legend in the subplot at position (0, 2)
        if plot_index == 2:
            ax.legend(loc='upper right')

        # Move to the next subplot
        plot_index += 1

    # Add title to whole figure
    fig.suptitle(f'{method}: Rate of KPIs at Different Velocities', fontsize=16)

    # Adjust the layout for better fit and show the plot
    plt.tight_layout()
    plt.show()


# Loop through each method
for method in loop_methods:
    # Round the SD Threshold values to 5 decimal places and then filter
    multi_run_stats_df['SD Threshold'] = multi_run_stats_df['SD Threshold'].round(5)

    # Filter the DataFrame for the current method and the desired threshold
    filtered_df = multi_run_stats_df[(multi_run_stats_df['Method'] == method) &
                                     (multi_run_stats_df['SD Threshold'] == 4.00000)]

    # Filter out all Total velocities
    filtered_df = filtered_df[filtered_df['Velocity'] != 'Total']

    # Define the KPI list based on the column names
    kpis = ['A: True Spike, True Velocity', 'B: True Spike, Wrong Velocity',
            'C: True Spike, Wrong Velocity (Infinite Velocity)', 'D: Missed Spike']

    # Mapping KPI codes to full names
    kpi_names = {
        'A: True Spike, True Velocity': "A: True Spike,\nTrue Velocity",
        'B: True Spike, Wrong Velocity': "B: True Spike,\nWrong Velocity",
        'C: True Spike, Wrong Velocity (Infinite Velocity)': "C: True Spike,\nWrong Velocity\n(Infinite Velocity)",
        'D: Missed Spike': "D: Missed\nSpike"
    }

    # Reshape the DataFrame using melt
    melted_df = filtered_df.melt(id_vars=['Velocity'], value_vars=kpis, var_name='KPI', value_name='Value')

    # Pivot the DataFrame to organize data for plotting
    plot_data = melted_df.pivot_table(index='KPI', columns='Velocity', values='Value')

    # Create a figure for the bar chart
    fig, ax = plt.subplots(figsize=(7, 11))

    # Plot each Velocity as a separate group of bars
    bar_width = 0.15
    positions = range(len(plot_data.index))

    # Define the labels for the velocities
    velocity_labels = {v: f"{v} m/s" for v in plot_data.columns}

    for i, velocity in enumerate(plot_data.columns):
        bar_positions = [p + i * bar_width for p in positions]
        ax.bar(bar_positions, plot_data[velocity].values, width=bar_width, label=velocity_labels[velocity], alpha=0.6,
               edgecolor='black')

    # Set the x-ticks to be in the center of the grouped bars
    ax.set_xticks([p + bar_width * (len(plot_data.columns) / 2) for p in positions])
    ax.set_xticklabels([kpi_names[kpi] for kpi in plot_data.index])
    ax.set_ylim(0, 0.7)

    # Set labels and title
    ax.set_ylabel('Rate')
    ax.legend(title='Velocity', loc='upper right', bbox_to_anchor=(1, 1), ncol=2)

    # Adjust layout for better fit
    plt.tight_layout()
    plt.title(f'{method}: Rate of KPIs at σ = 4.0')

    # Show the plot
    plt.show()


# Round the SD Threshold values to 5 decimal places and then filter
multi_run_stats_df['SD Threshold'] = multi_run_stats_df['SD Threshold'].round(5)

# Filter the DataFrame for the desired velocity and threshold
filtered_df = multi_run_stats_df[(multi_run_stats_df['Velocity'] == 'Total') &
                                 (multi_run_stats_df['SD Threshold'] == 4.00000)]

# Define the KPI list based on the column names
kpis = ['A: True Spike, True Velocity', 'B: True Spike, Wrong Velocity',
        'C: True Spike, Wrong Velocity (Infinite Velocity)', 'D: Missed Spike', 'E: False Spike']

# Create a figure for the bar chart
fig, ax = plt.subplots(figsize=(8, 5))

# Reshape the DataFrame using melt and pivot
melted_df = filtered_df.melt(id_vars=['Method'], value_vars=kpis, var_name='KPI', value_name='Value')

# Ensure the methods are sorted in the desired order
method_order = ['AHC', 'SINC', 'SPLINE', 'CORR']
melted_df['Method'] = pd.Categorical(melted_df['Method'], categories=method_order, ordered=True)
plot_data = melted_df.pivot_table(index='KPI', columns='Method', values='Value')

# Mapping method codes to full names
method_names = {
    'AHC': "Alternating Hill Climbing",
    'SINC': "Sinc Interpolation",
    'SPLINE': "Cubic Spline Interpolation",
    'CORR': "Cross-Correlation"
}

# Mapping KPI codes to full names
kpi_names = {
    'A: True Spike, True Velocity': "A: True Spike,\nTrue Velocity",
    'B: True Spike, Wrong Velocity': "B: True Spike,\nWrong Velocity",
    'C: True Spike, Wrong Velocity (Infinite Velocity)': "C: True Spike,\nWrong Velocity\n(Infinite Velocity)",
    'D: Missed Spike': "D: Missed\nSpike",
    'E: False Spike': "E: False\nSpike",
}

# Plot each Method as a separate group of bars
bar_width = 0.15
positions = np.arange(len(plot_data.index))

for i, method in enumerate(method_order):
    bar_positions = positions + i * bar_width
    ax.bar(bar_positions, plot_data[method], width=bar_width, label=method_names[method], alpha=0.6, edgecolor='black')

# Set the x-ticks to be in the center of the grouped bars
ax.set_xticks(positions + bar_width * (len(plot_data.columns) / 2 - 0.5))
ax.set_xticklabels([kpi_names[kpi] for kpi in plot_data.index])
ax.set_ylim(0, 0.7)

# Set labels and title
ax.set_ylabel('Rate')
ax.legend()

# Adjust layout for better fit
plt.title('Rate of KPIs at σ = 4.0 for All Methods')
plt.tight_layout()
plt.show()


for method in loop_methods:
    # Calculate the current method's Directional Percentage Error
    multiple_run_signal_df[f'{method} Directional Percentage Error'] = (
            ((multiple_run_signal_df[f'{method} Detected Velocity'] - multiple_run_signal_df['True Velocity']) /
             abs(multiple_run_signal_df['True Velocity'])) * 100
    )

    # Extract the error values
    error_values = multiple_run_signal_df[f'{method} Directional Percentage Error']

    # Count occurrences of infinite values (both positive and negative)
    inf_count = np.isinf(error_values).sum()

    # Filter finite values for binning
    finite_values = error_values[np.isfinite(error_values)]

    # Define the bins and labels
    bins = [-np.inf, -1000, -100, -40, -5, 5, 40, 100, 1000, np.inf]
    labels = [
        '< -1000%',
        '-1000%\nto\n-100%',
        '-100%\nto\n-40%',
        '-40%\nto\n-5%',
        '-5%\nto\n5%',
        '5%\nto\n40%',
        '40%\nto\n100%',
        '100%\nto\n1000%',
        '>1000%'
    ]

    # Bin the finite error values
    finite_bins = pd.cut(finite_values, bins=bins, labels=labels, right=False, include_lowest=True)

    # Count the occurrences in each bin
    finite_counts = finite_bins.value_counts().reindex(labels, fill_value=0)

    # Create a DataFrame for finite counts
    finite_df = pd.DataFrame({
        'Error Region': labels,
        'Count': finite_counts.values
    })

    # Create a DataFrame for infinite counts
    infinite_df = pd.DataFrame({
        'Error Region': ['∞'],
        'Count': [inf_count]
    })

    # Concatenate the infinite and finite counts
    error_region_counts = pd.concat([infinite_df, finite_df], ignore_index=True)

    # Define the desired order of Error Regions
    error_regions_order = [
        '< -1000%',
        '-1000%\nto\n-100%',
        '-100%\nto\n-40%',
        '-40%\nto\n-5%',
        '-5%\nto\n5%',
        '5%\nto\n40%',
        '40%\nto\n100%',
        '100%\nto\n1000%',
        '>1000%',
        '∞'
    ]

    # Reindex the DataFrame to match the desired order
    error_region_counts.set_index('Error Region', inplace=True)
    error_region_counts = error_region_counts.reindex(error_regions_order).reset_index()

    # Recalculate total_rows to include infinite values
    total_rows = multiple_run_signal_df.shape[0]

    # Calculate the percentage of total rows for each region
    error_region_counts['Percentage'] = (error_region_counts['Count'] / total_rows) * 100

    # Define colors for each bar based on the specified regions
    colors = []
    for region in error_regions_order:
        if region == '∞':
            colors.append('lightblue')  # Dark red for infinity
        elif region in ['< -1000%', '-1000%\nto\n-100%']:
            colors.append('lightcoral')  # Light red for below -100%
        elif region in ['-40%\nto\n-5%', '-5%\nto\n5%', '5%\nto\n40%']:
            colors.append('green')  # Green for -40% to 40%
        else:
            colors.append('yellow')  # Yellow for everything else

    # Define labels with percentages for the legend
    green_label = f"True Velocity ({error_region_counts.loc[error_region_counts['Error Region'].isin(['-40%\nto\n-5%', '-5%\nto\n5%', '5%\nto\n40%']), 'Percentage'].sum():.2f}%)"
    yellow_label = f"Wrong Velocity, Correct Direction ({error_region_counts.loc[error_region_counts['Error Region'].isin(['-100%\nto\n-40%', '40%\nto\n100%', '100%\nto\n1000%', '>1000%']), 'Percentage'].sum():.2f}%)"
    light_red_label = f"Wrong Velocity, Wrong Direction ({error_region_counts.loc[error_region_counts['Error Region'].isin(['< -1000%', '-1000%\nto\n-100%']), 'Percentage'].sum():.2f}%)"
    dark_red_label = f"Wrong Velocity, Infinite Velocity ({error_region_counts.loc[error_region_counts['Error Region'] == '∞', 'Percentage'].sum():.2f}%)"

    # Plotting the percentages with custom colors and no rotation
    plt.figure(figsize=(14, 7))
    plt.bar(
        error_region_counts['Error Region'],
        error_region_counts['Percentage'],
        color=colors,
        edgecolor='black'
    )
    plt.xlabel(f'{method} Directional Percentage Error Regions', fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.title(f'{method}: Percentage of Directional Percentage Error', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Create custom legend
    plt.legend(handles=[
        plt.Line2D([0], [0], color='green', lw=10, label=green_label),
        plt.Line2D([0], [0], color='yellow', lw=10, label=yellow_label),
        plt.Line2D([0], [0], color='lightcoral', lw=10, label=light_red_label),
        plt.Line2D([0], [0], color='lightblue', lw=10, label=dark_red_label)
    ], loc='upper right', fontsize=12)

    plt.title(f'{method}: Directional Percentage Error', fontsize=16)
    plt.tight_layout()
    plt.show()


for method in loop_methods:
    # Set up the plot grid: 3 rows, 2 columns for each method
    fig, axs = plt.subplots(3, 2, figsize=(18, 12), sharex=True, sharey=True)
    axs = axs.flatten()  # Flatten the array of axes for easy iteration

    for i, signal_speed in enumerate(signal_speeds):
        if i >= len(axs):
            break  # Stop if there are no more subplots available

        # Filter the dataframe for the current absolute ground truth velocity
        signal_df = multiple_run_signal_df[np.abs(multiple_run_signal_df['True Velocity']) == signal_speed]

        # Calculate the current method's Directional Percentage Error
        signal_df[f'{method} Directional Percentage Error'] = (
                ((signal_df[f'{method} Detected Velocity'] - signal_df['True Velocity']) /
                 abs(signal_df['True Velocity'])) * 100
        )

        # Extract the error values
        error_values = signal_df[f'{method} Directional Percentage Error']

        # Count occurrences of infinite values (both positive and negative)
        inf_count = np.isinf(error_values).sum()

        # Filter finite values for binning
        finite_values = error_values[np.isfinite(error_values)]

        # Define the bins and labels
        bins = [-np.inf, -1000, -100, -40, -5, 5, 40, 100, 1000, np.inf]
        labels = [
            '< -1000%',
            '-1000%\nto\n-100%',
            '-100%\nto\n-40%',
            '-40%\nto\n-5%',
            '-5%\nto\n5%',
            '5%\nto\n40%',
            '40%\nto\n100%',
            '100%\nto\n1000%',
            '>1000%'
        ]

        # Bin the finite error values
        finite_bins = pd.cut(finite_values, bins=bins, labels=labels, right=False, include_lowest=True)

        # Count the occurrences in each bin
        finite_counts = finite_bins.value_counts().reindex(labels, fill_value=0)

        # Create a DataFrame for finite counts
        finite_df = pd.DataFrame({
            'Error Region': labels,
            'Count': finite_counts.values
        })

        # Create a DataFrame for infinite counts
        infinite_df = pd.DataFrame({
            'Error Region': ['∞'],
            'Count': [inf_count]
        })

        # Concatenate the infinite and finite counts
        error_region_counts = pd.concat([infinite_df, finite_df], ignore_index=True)

        # Define the desired order of Error Regions
        error_regions_order = [
            '< -1000%',
            '-1000%\nto\n-100%',
            '-100%\nto\n-40%',
            '-40%\nto\n-5%',
            '-5%\nto\n5%',
            '5%\nto\n40%',
            '40%\nto\n100%',
            '100%\nto\n1000%',
            '>1000%',
            '∞'
        ]

        # Reindex the DataFrame to match the desired order
        error_region_counts.set_index('Error Region', inplace=True)
        error_region_counts = error_region_counts.reindex(error_regions_order).reset_index()

        # Recalculate total_rows to include infinite values
        total_rows = signal_df.shape[0]

        # Calculate the percentage of total rows for each region
        error_region_counts['Percentage'] = (error_region_counts['Count'] / total_rows) * 100

        # Define colors for each bar based on the specified regions
        colors = []
        for region in error_regions_order:
            if region == '∞':
                colors.append('lightblue')  # Light blue for infinity
            elif region in ['< -1000%', '-1000%\nto\n-100%']:
                colors.append('lightcoral')  # Light red for below -100%
            elif region in ['-40%\nto\n-5%', '-5%\nto\n5%', '5%\nto\n40%']:
                colors.append('green')  # Green for -40% to 40%
            else:
                colors.append('yellow')  # Yellow for everything else

        # Define labels with percentages for the legend
        green_label = f"True Velocity ({error_region_counts.loc[error_region_counts['Error Region'].isin(['-40%\nto\n-5%', '-5%\nto\n5%', '5%\nto\n40%']), 'Percentage'].sum():.1f}%)"
        yellow_label = f"Wrong Velocity, Correct Direction ({error_region_counts.loc[error_region_counts['Error Region'].isin(['-100%\nto\n-40%', '40%\nto\n100%', '100%\nto\n1000%', '>1000%']), 'Percentage'].sum():.1f}%)"
        light_red_label = f"Wrong Velocity, Wrong Direction ({error_region_counts.loc[error_region_counts['Error Region'].isin(['< -1000%', '-1000%\nto\n-100%']), 'Percentage'].sum():.1f}%)"
        dark_red_label = f"Wrong Velocity, Infinite Velocity ({error_region_counts.loc[error_region_counts['Error Region'] == '∞', 'Percentage'].sum():.1f}%)"

        # Plotting the percentages in the specified subplot
        axs[i].bar(
            error_region_counts['Error Region'],
            error_region_counts['Percentage'],
            color=colors,
            edgecolor='black'
        )
        axs[i].set_xlabel(f'{method} Error Regions', fontsize=12)
        axs[i].set_ylabel('Percentage (%)', fontsize=12)
        axs[i].set_title(f'{method} Error for |Velocity| = {signal_speed}', fontsize=14)

        # Add legend to each subplot
        axs[i].legend(handles=[
            plt.Line2D([0], [0], color='green', lw=10, label=green_label),
            plt.Line2D([0], [0], color='yellow', lw=10, label=yellow_label),
            plt.Line2D([0], [0], color='lightcoral', lw=10, label=light_red_label),
            plt.Line2D([0], [0], color='lightblue', lw=10, label=dark_red_label)
        ], loc='upper right', fontsize=10)

    # Only show y and x labels on the left and bottom subplots
    for ax in axs:
        ax.label_outer()

    # Set the main title for the method
    plt.suptitle(f'{method}: Directional Percentage Error across Absolute True Velocities', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit the title and legend
    plt.show()