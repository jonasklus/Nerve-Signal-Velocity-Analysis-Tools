# Nerve Signal Velocity Analysis Tools
### Klus et al., 2024

## Synthetic nerve signal simulation classes
Python classes to simulate and analyze nerve signals and their propagation through a cuff electrode.
- **synthetic_data_simulation.py** - Contains classes to simulate nerve signals and their propagation through a cuff electrode.
- **isi_analysis.py** - Contains interpolation and velocity analysis methods for simulated nerve signals.
- **multi_run_stats.py** - Contains methods to run multiple simulations, outputting the aggregated statistics and the simulated data.

## Data analysis tools
Python scripts to analyze and plot the data from the simulations.
- **cross_corr.py** - Runs cross-correlation analysis on the simulated data and plots results.
- **velocity_methods.py** - Runs the comparative velocity analysis on the simulated data and plots results.
- **sampling_limitations.py** - Plots the explanatory figures for the sampling limitations of the velocity analysis.

## In vivo electrophysiology validation
Matlab code to filter and analyse electrophysiology recordings from ultraconformable cuffs for in vivo validation of nerve signal velocity analysis tools.
- **read_Intan_RHS2000_file.m** - Imports recording data from intan data file format (.rhs) to matlab. Provided by Intan Technlogies (https://intantech.com/downloads.html?tabSelect=Software&yPos=100).
- **Processed_bipolar_MEA_Intan.m** - Bandpass filters and performs referencing of imported recordings.
- **crosscoeff_plot.m** - Carries out a cross-correlation calculation between two channel recordings from the in vivo data and plots the results.
