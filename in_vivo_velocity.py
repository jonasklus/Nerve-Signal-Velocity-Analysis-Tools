
# Import libraries
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Load the mat file
full_data = loadmat('./spike_interval_processed.mat')
data = np.array(full_data['Processed_data'])

# Split the data into first row (proximal) the second row (distal)
signal1 = data[0, :]
signal2 = data[1, :]


# Define function to determine the peak velocity
def determine_peak_velocity(signal1, signal2, sampling_rate, sd_threshold, sample_window, signal_type):
    # Create velocity_df with columns Method, Peak_1, Peak_2, Delay_ms, Velocity
    velocity_df = pd.DataFrame(columns=['Method', 'Spike_1', 'Spike_2', 'Peak Delay ms', 'Peak Velocity'])

    # Calculate amplitude corresponding to the threshold
    sd_window_start = 0 * sampling_rate
    sd_window_end = 2 * sampling_rate
    sd_amp = np.std(signal1[sd_window_start:sd_window_end])
    sd_amp_threshold = sd_amp * sd_threshold

    # Initialize variables
    curr_peak_signal_1 = np.nan
    curr_peak_signal_2 = np.nan
    curr_delay_ms = np.nan
    curr_velocity = np.nan

    # Create shifted versions of signal_1 for comparison
    shifted_left = np.roll(signal1, 1)
    shifted_right = np.roll(signal1, -1)

    # Create a boolean array where the peaks are located
    peaks_mask = (signal1 < -sd_amp_threshold) & (signal1 < shifted_left) & (signal1 < shifted_right)

    # Use np.where to get the indices of these peaks
    peak_indices = np.where(peaks_mask)[0]

    for i in peak_indices:
        curr_peak_signal_1 = i
        curr_peak_signal_2 = np.nan
        # Crawl through signal 2 to find the first spike above the threshold around the peak in signal 1
        for j in range(sample_window):
            # Check if j is the first or last index
            if i + j <= 0 or i + j >= len(signal2) - 1 or i - j <= 0 or i - j >= len(signal2) - 1:
                continue

            # First check if there is a peak in the positive direction i + j
            if (signal2[i + j] < -sd_amp_threshold and
                    signal2[i + j] < signal2[i + j - 1] and
                    signal2[i + j] < signal2[i + j + 1]):
                curr_peak_signal_2 = i + j
                break
            # Then check if there is a peak in the negative direction i - j
            elif (signal2[i - j] < -sd_amp_threshold and
                  signal2[i - j] < signal2[i - j - 1] and
                  signal2[i - j] < signal2[i - j + 1]):
                curr_peak_signal_2 = i - j
                break

        # Calculate the velocity of the detected peak if both peaks are found
        curr_delay_ms = np.nan
        curr_velocity = np.nan
        if np.isnan(curr_peak_signal_1) == False and np.isnan(curr_peak_signal_2) == False:
            if curr_peak_signal_1 == curr_peak_signal_2:
                curr_velocity = np.inf
                curr_delay_ms = 0
            else:
                # Calculate the delay in ms
                curr_delay_ms = ((curr_peak_signal_2 - curr_peak_signal_1) / sampling_rate) * 1000

                # Calculate the velocity
                distance_m = 0.002  # 2 mm in meters
                curr_velocity = distance_m / (curr_delay_ms / 1000)  # Convert to meters per second

        # Convert spike 1 to float
        curr_peak_signal_1 = float(curr_peak_signal_1)

        # Append the values to the velocity_df
        velocity_df.loc[len(velocity_df)] = [signal_type, curr_peak_signal_1, curr_peak_signal_2, curr_delay_ms,
                                             curr_velocity]

    # Return the velocity_df
    return velocity_df

# Define function to determine the cross-correlation velocity
def determine_cross_coeff_velocity(signal1, signal2, sampling_rate, velocity_df):
    # Determine scale factor for the sample window
    df_search_scaler = int(sampling_rate / 30000)

    # Create columns in the signal dataframe
    velocity_df[f'CrossCoeff Delay ms'] = np.nan
    velocity_df[f'CrossCoeff Velocity'] = np.nan

    # Loop through each row with iterrows
    for index, row in velocity_df.iterrows():
        # Initialize variables
        delay_ms_cross_coeff = np.nan
        velocity_cross_coeff = np.nan

        # Check if AHC Detected Peak 1 and AHC Detected Peak 2 are not NaN
        if not np.isnan(row['Spike_1']) and not np.isnan(row['Spike_2']):
            # Run a cross-correlation between the two signals 100 samples before and after the detected peaks in both directions
            start = int(row['Spike_1']) - (100 * df_search_scaler)
            end = int(row['Spike_1']) + (100 * df_search_scaler)
            s1 = signal1[start:end]
            s2 = signal2[start:end]

            # Calculate the cross-correlation between the two signals
            corr_true = signal.correlate(s2, s1, "full")

            # Find the maximum correlation value and its index
            max_pos_true = np.argmax(corr_true)

            # Find the maximum cross-correlation value and calculate the velocity
            sig_delay_samples = max_pos_true - ((100 * df_search_scaler) * 2 - 1)

            # Calculate the delay in ms and the velocity
            # Calculate the velocity
            distance_m = 0.002  # 2 mm in meters
            delay_ms_cross_coeff = sig_delay_samples / sampling_rate * 1000
            if delay_ms_cross_coeff != 0:
                velocity_cross_coeff = ((distance_m * sampling_rate * df_search_scaler) / sig_delay_samples)
            else:
                velocity_cross_coeff = np.inf

            # Assign the calculated velocity to the spike dataframe
            velocity_df.at[index, f'CrossCoeff Delay ms'] = delay_ms_cross_coeff
            velocity_df.at[index, f'CrossCoeff Velocity'] = velocity_cross_coeff

    return velocity_df

# Define function to interpolate the signal with spline
def spline_interpolate(input_signal, upsample_factor):
    # Create a time array
    time = np.arange(0, len(input_signal))

    # Create a time array for the upsampled signal
    upsampled_time = np.linspace(0, len(input_signal), len(input_signal) * upsample_factor)

    # Create a spline interpolation object
    spline = CubicSpline(time, input_signal)

    # Interpolate the signal
    interpolated_signal = spline(upsampled_time)

    return interpolated_signal

# Define function to interpolate the signal with sinc
def sinc_interpolate(input_signal, upsample_factor):
    # Define parameters
    num_output = len(input_signal) * upsample_factor
    x = input_signal
    s = input_signal.shape[0]

    # Compute the FFT of the input signal
    X = np.fft.rfft(x)

    # Create a new array for the zero-padded frequency spectrum
    X_padded = np.zeros(num_output // 2 + 1, dtype=complex)

    # Copy the original frequency spectrum into the zero-padded array
    X_padded[:X.shape[0]] = X

    # Compute the inverse FFT of the zero-padded frequency spectrum
    x_interpolated = np.fft.irfft(X_padded, n=num_output)

    # Create output signal
    output_signal = x_interpolated * (num_output / s)

    return output_signal

# Interpolate the signals
upsample_factor = 100

# Spline interpolate the signals
signal1_spline = spline_interpolate(signal1, upsample_factor)
signal2_spline = spline_interpolate(signal2, upsample_factor)

# Sinc interpolate the signals
signal1_sinc = sinc_interpolate(signal1, upsample_factor)
signal2_sinc = sinc_interpolate(signal2, upsample_factor)


# Set the SD threshold
sd_threshold = [4]
upsample_factor = 100

# Calculate total number of iterations and new run bool
new_run = True

for sd_threshold in sd_threshold:
    for loop_signal in ['sampled', 'spline', 'sinc']:
        # Reset velocity_df
        velocity_df = pd.DataFrame(columns=['Method', 'Spike_1', 'Spike_2', 'Peak Delay ms', 'Peak Velocity'])

        if loop_signal == 'sampled':
            # De
            sampling_rate = 30000
            sample_window = 100
            signal_type = 'Sampled Signal'

            # Load the signals
            sig1 = signal1.copy()
            sig2 = signal2.copy()

        elif loop_signal == 'spline':
            # Define parameters
            sampling_rate = 30000 * upsample_factor
            sample_window = 100 * upsample_factor
            signal_type = 'Spline Interpolated Signal'

            # Load the signals
            sig1 = signal1_spline.copy()
            sig2 = signal2_spline.copy()

        elif loop_signal == 'sinc':
            # Define parameters
            sampling_rate = 30000 * upsample_factor
            sample_window = 100 * upsample_factor
            signal_type = 'Sinc Interpolated Signal'

            # Load the signals
            sig1 = signal1_sinc.copy()
            sig2 = signal2_sinc.copy()

        # Run the function
        velocity_df = determine_peak_velocity(sig1, sig2, sampling_rate, sd_threshold, sample_window, signal_type)
        velocity_df = determine_cross_coeff_velocity(sig1, sig2, sampling_rate, velocity_df)

        # Add the sd_threshold as column to the velocity_df and sort columns
        velocity_df['SD Threshold'] = sd_threshold
        velocity_df = velocity_df[
            ['SD Threshold', 'Method', 'Spike_1', 'Spike_2', 'Peak Delay ms', 'Peak Velocity', 'CrossCoeff Delay ms',
             'CrossCoeff Velocity']]

        # If new run, create full_velocity_df as copy of velocity_df, else append to full_velocity_df
        if new_run:
            full_velocity_df = velocity_df.copy()
            new_run = False
        else:
            full_velocity_df = pd.concat([full_velocity_df, velocity_df], ignore_index=True)



# Set all fonts to Times New Roman and size 10
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'font.size': 18
})

# Load the signals
trace_1 = signal1.copy()
trace_2 = signal2.copy()

# Keep only the non-NaN values in Spike_1
velocity_df_s2 = determine_peak_velocity(trace_2, trace_1, 30000, 4, 100, 'Sampled Signal')
velocity_df_s2 = velocity_df_s2.dropna(subset=['Spike_1'])

# Plot the signal with detected spikes
plt.figure(figsize=(12, 8))
plt.plot(trace_2, color='#FFA500')
plt.ylabel('Distal')
plt.xlim(0, len(trace_2))
plt.ylim(-100, 100)
plt.xticks([])
plt.yticks([])
plt.box(False)
for spike in velocity_df_s2["Spike_1"]:
    plt.axvline(x=spike, color='black', ymax=0.15, linewidth=0.5)
plt.show()

# Keep only the non-NaN values in Spike_1
velocity_df_s1 = determine_peak_velocity(trace_1, trace_2, 30000, 4, 100, 'Sampled Signal')
velocity_df_s1 = velocity_df_s1.dropna(subset=['Spike_1'])

# Plot the signal with detected spikes
plt.figure(figsize=(12, 8))
plt.plot(trace_1, color='#2572BD')
plt.ylabel('Proximal')
plt.xlim(0, len(trace_1))
plt.ylim(-100, 100)
plt.xticks([])
plt.yticks([])
plt.box(False)
for spike in velocity_df_s1["Spike_1"]:
    plt.axvline(x=spike, color='black', ymax=0.15, linewidth=0.5)
plt.show()

# Keep only the non-NaN values in Spike_1
velocity_df_full = determine_peak_velocity(trace_1, trace_2, 30000, 4, 100, 'Sampled Signal')
velocity_df_full = velocity_df_full.dropna(subset=['Spike_2'])

# Plot the signal with detected spikes
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(trace_2, color='#FFA500')
plt.ylabel('Distal')
plt.xlim(0, len(trace_2))
plt.ylim(-100, 100)
plt.xticks([])
plt.yticks([])
plt.box(False)
for spike in velocity_df_full["Spike_2"]:
    plt.axvline(x=spike, color='black', ymax=0.15, linewidth=0.5)

plt.subplot(2, 1, 2)
plt.plot(trace_1, color='#2572BD')
plt.ylabel('Proximal')
plt.xlim(0, len(trace_1))
plt.ylim(-100, 100)
plt.xticks([])
plt.yticks([])
plt.box(False)
for spike in velocity_df_full["Spike_1"]:
    plt.axvline(x=spike, color='black', ymax=0.15, linewidth=0.5)
plt.show()


# Define parameters pairs (Method and Metric column)
method_metric_pairs = [
    ['Sampled Signal', 'Peak Delay ms'],
    ['Sinc Interpolated Signal', 'Peak Delay ms'],
    ['Spline Interpolated Signal', 'Peak Delay ms'],
    ['Sampled Signal', 'CrossCoeff Delay ms']]

# Set up histogram parameters
bin_size = 0.1  # Bin size remains 0.1
bins = [round(x * bin_size - 0.05, 2) for x in range(-20, 21)]  # Shift bins by -0.05, ranging from -2.05 to 2.05 ms

# Plot the histograms as individual plots
for method, metric in method_metric_pairs:
    # Filter the full_velocity_df by method and metric
    filtered_df = full_velocity_df[full_velocity_df['Method'] == method]
    filtered_df = filtered_df[filtered_df['SD Threshold'] == 4]

    # Define names
    if metric == 'Peak Delay ms':
        metric_name = 'Peak velocity'
    elif metric == 'CrossCoeff Delay ms':
        metric_name = 'CrossCoeff velocity'

    if method == 'Sampled Signal':
        method_Name = 'Sampled signal'
    elif method == 'Sinc Interpolated Signal':
        method_Name = 'Sinc interpolation'
    elif method == 'Spline Interpolated Signal':
        method_Name = 'Spline interpolation'

    # Plot the histogram
    plt.figure(figsize=(6, 8))
    plt.hist(filtered_df[metric], bins=bins, color='skyblue', edgecolor='black')

    # Add a red bar at 0 for all delays that are exactly 0
    zero_count = (filtered_df[metric] == 0).sum()
    plt.bar(0, zero_count, width=0.1, color='red', edgecolor='black')
    plt.xlabel('Delay (ms)')
    plt.ylabel('Spike Count')
    plt.xticks(np.arange(-2, 2.1, 1))
    plt.axvline(0, color='grey', linestyle=':')
    plt.xlim(-2, 2)
    plt.title(f'{method_Name} + {metric_name}')
    plt.show()
