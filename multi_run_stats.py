
# Import libraries
import tqdm
import copy
import numpy as np
import pandas as pd
from synthetic_data_simulation import SyntheticPropagatingNerveSignal
from isi_analysis import InterSpikeIntervallAnalyzer


class MultiRunStatsCreator:
    """
    This class creates statistics for multiple runs of synthetic nerve signals
    with different standard deviation thresholds and velocities. The statistics
    include the rates of true spikes with true velocities, true spikes with
    wrong velocities, true spikes with wrong velocities (infinite velocity),
    missed spikes, and false spikes.
    """
    # Constructor
    def __init__(self, runs_per_sd, sd_thresholds, velocities):
        self.runs_per_sd = runs_per_sd
        self.sd_thresholds = sd_thresholds
        self.velocities = velocities
        self.multiple_run_signal_df = pd.DataFrame()
        self.statistics_df = pd.DataFrame()

        # Run multiple simulations
        self.run_multiple_simulations()

        # Create statistics
        self.create_statistics()


    # Define function to run multiple simulations
    def run_multiple_simulations(self):
        # Set create_new_df marker to True
        create_new_df = True

        # Loop through runs per standard deviation threshold
        for run in tqdm.tqdm(range(self.runs_per_sd)):
            # Create synthetic nerve signal
            nerve_signal = SyntheticPropagatingNerveSignal(self.velocities)

            # Loop through runs per standard deviation threshold
            for sd_threshold in self.sd_thresholds:
                # Copy nerve signal to avoid overwriting
                nerve_signal_copy = copy.deepcopy(nerve_signal)

                # Create inter spike interval analyzer
                analyzer = InterSpikeIntervallAnalyzer(nerve_signal_copy, sd_threshold)

                # Add new columns "SD Threshold" and "Run" to signal data frame and bring to front
                analyzer.nerve_signal.signal_df.insert(0, "SD Threshold", sd_threshold)
                analyzer.nerve_signal.signal_df.insert(1, "Run", run + 1)

                # Create new df if marker is set, else concat to existing df
                if create_new_df:
                    self.multiple_run_signal_df = analyzer.nerve_signal.signal_df.copy()
                    create_new_df = False
                else:
                    self.multiple_run_signal_df = pd.concat([self.multiple_run_signal_df, analyzer.nerve_signal.signal_df],
                                                            axis=0, ignore_index=True)


    # Define function to create statistics
    def create_statistics(self):
        # Create columns for statistics data frame
        self.statistics_df = pd.DataFrame(columns=["Method", "Velocity", "SD Threshold",
                                                    "A: True Spike, True Velocity", "B: True Spike, Wrong Velocity",
                                                    "C: True Spike, Wrong Velocity (Infinite Velocity)",
                                                    "D: Missed Spike", "E: False Spike"])

        # Add "Total" to velocity list at first position and create new list
        loop_velocities = self.velocities.copy()
        loop_velocities.insert(0, "Total")

        # Loop through methods
        method_list = ["AHC", "SINC", "SPLINE", "CORR"]
        for method in method_list:

            # Loop through velocities
            for velocity in loop_velocities:

                # Loop through standard deviation thresholds
                for sd_threshold in self.sd_thresholds:
                    # Get filtered signal data frame
                    if velocity == "Total":
                        filtered_df = self.multiple_run_signal_df[self.multiple_run_signal_df["SD Threshold"] == sd_threshold]
                    else:
                        filtered_df = self.multiple_run_signal_df[(self.multiple_run_signal_df["True Velocity"] == velocity) &
                                                                  (self.multiple_run_signal_df["SD Threshold"] == sd_threshold)]

                    # Create column "Total Spike" and set to 1 if "True Velocity" is not NaN
                    filtered_df["Total Spike"] = np.where(filtered_df["True Velocity"].notna(), 1, 0)
                    filtered_df["A: True Spike, True Velocity"] = np.where((filtered_df[f"{method} Correct Velocity Detected"] == True), 1, 0)
                    filtered_df["B: True Spike, Wrong Velocity"] = np.where((filtered_df[f"{method} Correct Velocity Detected"] == False), 1, 0)
                    filtered_df["C: True Spike, Wrong Velocity (Infinite Velocity)"] = np.where((filtered_df[f"{method} Correct Velocity Detected"] == False) &
                                                                                             (filtered_df[f"{method} Detected Velocity"] == np.inf), 1, 0)
                    filtered_df["D: Missed Spike"] = np.where((filtered_df["True Velocity"].notna()) &
                                                              (filtered_df[f"{method} Correct Velocity Detected"].isna()), 1, 0)

                    if method == "AHC" or method == "SINC":
                        filtered_df["E: False Spike"] = np.where((filtered_df[f"{method} Actual Spike Detected"] == False), 1, 0)
                    else:
                        filtered_df["E: False Spike"] = np.where((filtered_df[f"AHC Actual Spike Detected"] == False), 1, 0)

                    # Calculate rates for each category
                    true_spike_true_vel_rate = filtered_df["A: True Spike, True Velocity"].sum() / filtered_df["Total Spike"].sum()
                    true_spike_wrong_vel_rate = filtered_df["B: True Spike, Wrong Velocity"].sum() / filtered_df["Total Spike"].sum()
                    true_spike_wrong_vel_inf_rate = filtered_df["C: True Spike, Wrong Velocity (Infinite Velocity)"].sum() / filtered_df["Total Spike"].sum()
                    missed_spike_rate = filtered_df["D: Missed Spike"].sum() / filtered_df["Total Spike"].sum()
                    false_spike_rate = filtered_df["E: False Spike"].sum() / filtered_df["Total Spike"].sum()

                    # Append statistics to statistics data frame
                    self.statistics_df.loc[len(self.statistics_df)] = [method, velocity, sd_threshold, true_spike_true_vel_rate,
                                                                      true_spike_wrong_vel_rate, true_spike_wrong_vel_inf_rate,
                                                                      missed_spike_rate, false_spike_rate]



