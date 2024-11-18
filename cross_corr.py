
# Import  libraries
import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
from synthetic_data_simulation import SingleExtraCellularSpike, SyntheticPropagatingNerveSignal


# Define the signal creator function
def signal_creator(spike_mult, length_multiplier=1):
    # Total number of samples in the base signal
    base_length = 15000 * 20  # 300,000 samples
    spike_signal_1 = np.zeros(base_length)
    spike_signal_2 = np.zeros(base_length)

    # Generate the time axis
    time_axis = np.linspace(0, 5 * 20 * length_multiplier, base_length * length_multiplier)

    # Create and resample the spike to make it narrower
    original_spike = SingleExtraCellularSpike(spike_mult).spike_signal
    resample_factor = 0.6  # Reduced to make spikes narrower
    new_spike_length = int(len(original_spike) * resample_factor)
    spike = signal.resample(original_spike, new_spike_length)

    # Define the number of spikes and calculate margins and spacing
    num_spikes = 10
    total_spike_duration = num_spikes * len(spike)
    remaining_space = base_length - total_spike_duration
    num_intervals = num_spikes + 1  # Spaces before, between, and after spikes
    spacing = remaining_space // num_intervals  # Equal spacing

    # Add spikes to spike_signal_1 and spike_signal_2
    for i in range(num_spikes):
        # Calculate the position for the current spike
        pos1 = spacing * (i + 1) + len(spike) * i - 3000
        # Ensure the spike fits within the signal
        if pos1 + len(spike) <= base_length:
            spike_signal_1[pos1: pos1 + len(spike)] += spike
        else:
            spike_signal_1[pos1:] += spike[:base_length - pos1]

        # Position for spike_signal_2 (shifted by 6,000 samples)
        pos2 = pos1 + 6000
        if pos2 + len(spike) <= base_length:
            spike_signal_2[pos2: pos2 + len(spike)] += spike
        else:
            spike_signal_2[pos2:] += spike[:base_length - pos2]

    # Repeat the signals according to the length multiplier
    if length_multiplier > 1:
        spike_signal_1 = np.tile(spike_signal_1, length_multiplier)
        spike_signal_2 = np.tile(spike_signal_2, length_multiplier)
        time_axis = np.linspace(0, 5 * 20 * length_multiplier, len(spike_signal_1))

    # Define the low-pass filter
    nyquist = 0.5 * 3000000  # Assuming a sampling rate of 3,000,000 Hz
    normal_cutoff = 1000 / nyquist
    b, a = signal.butter(5, normal_cutoff, btype='low', analog=False)

    # Generate noise signals and filter them
    np.random.seed(36)
    noise_1 = np.random.normal(0, 6, len(time_axis))
    np.random.seed(41)
    noise_2 = np.random.normal(0, 6, len(time_axis))

    noise_1 = signal.lfilter(b, a, noise_1)
    noise_2 = signal.lfilter(b, a, noise_2)

    # Normalize and scale the noise
    noise_1 = noise_1 / np.sqrt(np.mean(noise_1 ** 2)) * 6
    noise_2 = noise_2 / np.sqrt(np.mean(noise_2 ** 2)) * 6

    # Add noise to the spike signals
    full_signal_1 = spike_signal_1 + noise_1
    full_signal_2 = spike_signal_2 + noise_2

    # Return the generated signals
    return time_axis, full_signal_1, full_signal_2, spike_signal_1, spike_signal_2


# Create zero signal and spikes
zero_signal = np.zeros(15000 * 6)
time_axis = np.linspace(0, 5 * 6, len(zero_signal))

high_spike = SingleExtraCellularSpike(12)
low_spike = SingleExtraCellularSpike(6)

# Create signals
signal_a_e1 = zero_signal.copy()
signal_a_e2 = zero_signal.copy()
signal_b_e1 = zero_signal.copy()
signal_b_e2 = zero_signal.copy()
signal_c_e1 = zero_signal.copy()
signal_c_e2 = zero_signal.copy()
signal_d_e1 = zero_signal.copy()
signal_d_e2 = zero_signal.copy()

# Create noise with same length as the signal
nerve_signal = SyntheticPropagatingNerveSignal()
noise_length = (len(nerve_signal.a_noise_signal_1) / 1000) * 30
noise_1 = nerve_signal.a_noise_signal_1[50000:50000 + int(noise_length)]
noise_2 = nerve_signal.a_noise_signal_2[50000:50000 + int(noise_length)]

# Add one high spike at 15000 in e1 and 45000 in e2
signal_a_e1[15000:15000 + len(high_spike.spike_signal)] = low_spike.spike_signal
signal_a_e2[60000:60000 + len(high_spike.spike_signal)] = low_spike.spike_signal
signal_b_e1[15000:15000 + len(low_spike.spike_signal)] = low_spike.spike_signal
signal_b_e1[30000:30000 + len(low_spike.spike_signal)] = low_spike.spike_signal
signal_b_e2[45000:45000 + len(low_spike.spike_signal)] = low_spike.spike_signal
signal_b_e2[60000:60000 + len(low_spike.spike_signal)] = low_spike.spike_signal
signal_c_e1[15000:15000 + len(high_spike.spike_signal)] = high_spike.spike_signal
signal_c_e2[60000:60000 + len(high_spike.spike_signal)] = high_spike.spike_signal
signal_d_e1[15000:15000 + len(high_spike.spike_signal)] = high_spike.spike_signal
signal_d_e1[30000:30000 + len(high_spike.spike_signal)] = high_spike.spike_signal
signal_d_e2[45000:45000 + len(high_spike.spike_signal)] = high_spike.spike_signal
signal_d_e2[60000:60000 + len(high_spike.spike_signal)] = high_spike.spike_signal

# Run cross-correlation for each pair of signals
raw_cross_corr_a = signal.correlate(signal_a_e2, signal_a_e1, mode='full')
raw_cross_corr_b = signal.correlate(signal_b_e2, signal_b_e1, mode='full')
raw_cross_corr_c = signal.correlate(signal_c_e2, signal_c_e1, mode='full')
raw_cross_corr_d = signal.correlate(signal_d_e2, signal_d_e1, mode='full')
raw_cross_corr_noise = signal.correlate(noise_2, noise_1, mode='full')

# Calculate the correlation time axis
time_axis_corr = np.linspace(-5 * 6, 5 * 6, len(raw_cross_corr_a))

# Start figure
fig, axs = plt.subplots(2, 5, figsize=(20, 7))
xlim = 5 * 6

# Plot signals
signal_ylim_min = -50
signal_ylim_max = 40
axs[0, 0].plot(time_axis, signal_a_e1)
axs[0, 0].plot(time_axis, signal_a_e2, color='orange')
axs[0, 0].axhline(y=0, color='black', linewidth=1.5)
axs[0, 0].set_ylim(signal_ylim_min, signal_ylim_max)
axs[0, 0].set_xlim(0, xlim)
axs[0, 1].plot(time_axis, signal_b_e1)
axs[0, 1].plot(time_axis, signal_b_e2, color='orange')
axs[0, 1].axhline(y=0, color='black', linewidth=1.5)
axs[0, 1].set_ylim(signal_ylim_min, signal_ylim_max)
axs[0, 1].set_xlim(0, xlim)
axs[0, 2].plot(time_axis, signal_c_e1)
axs[0, 2].plot(time_axis, signal_c_e2, color='orange')
axs[0, 2].axhline(y=0, color='black', linewidth=1.5)
axs[0, 2].set_ylim(signal_ylim_min, signal_ylim_max)
axs[0, 2].set_xlim(0, xlim)
axs[0, 3].plot(time_axis, signal_d_e1)
axs[0, 3].plot(time_axis, signal_d_e2, color='orange')
axs[0, 3].axhline(y=0, color='black', linewidth=1.5)
axs[0, 3].set_ylim(signal_ylim_min, signal_ylim_max)
axs[0, 3].set_xlim(0, xlim)
axs[0, 4].plot(time_axis, noise_1, label='Recording Site 1')
axs[0, 4].plot(time_axis, noise_2, color='orange', label='Recording Site 2')
axs[0, 4].axhline(y=0, color='black', linewidth=1.5)
axs[0, 4].set_ylim(signal_ylim_min, signal_ylim_max)
axs[0, 4].set_xlim(0, xlim)
axs[0, 4].legend()
marker_length_x = 5  # 5 ms
marker_length_y = 10  # 10 μV
x_start = 0.1
y_start = -45
axs[0, 0].plot([x_start, x_start + marker_length_x], [y_start, y_start], 'k-', lw=1)
axs[0, 0].plot([x_start, x_start], [y_start, y_start + marker_length_y], 'k-', lw=1)
axs[0, 0].text(x_start + marker_length_x / 2, y_start - 3, '5 ms', ha='center', va='top', fontsize=8)
axs[0, 0].text(x_start - 0.2, y_start + marker_length_y / 2, '10 μV', ha='right', va='center', rotation='vertical',
               fontsize=8)

# Plot the raw cross-correlation
raw_ylim_min = -1500000
raw_ylim_max = 5000000
axs[1, 0].plot(time_axis_corr, raw_cross_corr_a, color='black')
axs[1, 0].set_ylim(raw_ylim_min, raw_ylim_max)
axs[1, 0].set_xlim(-xlim, xlim)
axs[1, 0].axvline(x=0, color='black', linewidth=0.5, linestyle='--')
axs[1, 0].axhline(y=0, color='black', linewidth=0.5, linestyle='--')
axs[1, 1].plot(time_axis_corr, raw_cross_corr_b, color='black')
axs[1, 1].set_ylim(raw_ylim_min, raw_ylim_max)
axs[1, 1].set_xlim(-xlim, xlim)
axs[1, 1].axvline(x=0, color='black', linewidth=0.5, linestyle='--')
axs[1, 1].axhline(y=0, color='black', linewidth=0.5, linestyle='--')
axs[1, 2].plot(time_axis_corr, raw_cross_corr_c, color='black')
axs[1, 2].set_ylim(raw_ylim_min, raw_ylim_max)
axs[1, 2].set_xlim(-xlim, xlim)
axs[1, 2].axvline(x=0, color='black', linewidth=0.5, linestyle='--')
axs[1, 2].axhline(y=0, color='black', linewidth=0.5, linestyle='--')
axs[1, 3].plot(time_axis_corr, raw_cross_corr_d, color='black')
axs[1, 3].set_ylim(raw_ylim_min, raw_ylim_max)
axs[1, 3].set_xlim(-xlim, xlim)
axs[1, 3].axvline(x=0, color='black', linewidth=0.5, linestyle='--')
axs[1, 3].axhline(y=0, color='black', linewidth=0.5, linestyle='--')
axs[1, 4].plot(time_axis_corr, raw_cross_corr_noise, color='black')
axs[1, 4].set_ylim(raw_ylim_min, raw_ylim_max)
axs[1, 4].set_xlim(-xlim, xlim)
axs[1, 4].axvline(x=0, color='black', linewidth=0.5, linestyle='--')
axs[1, 4].axhline(y=0, color='black', linewidth=0.5, linestyle='--')
marker_length_x = 5  # 5 ms
marker_length_y = 1000000  # 10 μV
x_start = -5 * 6
y_start = -1450000
axs[1, 0].plot([x_start, x_start + marker_length_x], [y_start, y_start], 'k-', lw=1)
axs[1, 0].plot([x_start, x_start], [y_start, y_start + marker_length_y], 'k-', lw=1)
axs[1, 0].text(x_start + marker_length_x / 2, y_start - 200000, '5 ms', ha='center', va='top', fontsize=8)
axs[1, 0].text(x_start - 0.5, y_start + marker_length_y / 2, '1e6', ha='right', va='center', rotation='vertical',
               fontsize=8)

# Get rid of the spines
for ax in axs.flatten():
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)

# Adjust the space between subplots
plt.subplots_adjust(wspace=0.2, hspace=0.02)
plt.show()


# Create three signals
time_axis, signal_a_1, signal_a_2, spike_signal_a_1, spike_signal_a_2 = signal_creator(12)
time_axis, signal_b_1, signal_b_2, spike_signal_b_1, spike_signal_b_2 = signal_creator(2)
time_axis, signal_c_1, signal_c_2, spike_signal_c_1, spike_signal_c_2 = signal_creator(0)

# Compute the cross-correlation
raw_cross_corr_a = signal.correlate(spike_signal_a_2, spike_signal_a_1)
raw_cross_corr_b = signal.correlate(signal_a_2, signal_a_1)
raw_cross_corr_c = signal.correlate(signal_b_2, signal_b_1)
raw_cross_corr_d = signal.correlate(signal_c_2, signal_c_1)

# Plot the signals
fig, axs = plt.subplots(3, 4, figsize=(16, 10))
signal_ylim_max = 30
signal_ylim_min = -70
signal_xlim_max = 100

# Plot spikes
axs[0, 0].plot(time_axis, spike_signal_a_1, linewidth=1)
axs[0, 0].plot(time_axis, spike_signal_a_2, color="orange", linewidth=1)
axs[0, 0].axhline(y=0, color="black", linewidth=1.5)
axs[0, 0].set_ylim(signal_ylim_min, signal_ylim_max)
axs[0, 0].set_xlim(0, signal_xlim_max)
axs[0, 0].text(0.5, 0.93, "Spike Scaling Factor = 12 (100 ms)", horizontalalignment='center', verticalalignment='center',
               transform=axs[0, 0].transAxes)
axs[0, 1].plot(time_axis, spike_signal_a_1, linewidth=1)
axs[0, 1].plot(time_axis, spike_signal_a_2, color="orange", linewidth=1)
axs[0, 1].axhline(y=0, color="black", linewidth=1.5)
axs[0, 1].set_ylim(signal_ylim_min, signal_ylim_max)
axs[0, 1].set_xlim(0, signal_xlim_max)
axs[0, 1].text(0.5, 0.93, "Spike Scaling Factor = 12 (100 ms)", horizontalalignment='center', verticalalignment='center',
               transform=axs[0, 1].transAxes)
axs[0, 2].plot(time_axis, spike_signal_b_1, linewidth=1)
axs[0, 2].plot(time_axis, spike_signal_b_2, color="orange", linewidth=1)
axs[0, 2].axhline(y=0, color="black", linewidth=1.5)
axs[0, 2].set_ylim(signal_ylim_min, signal_ylim_max)
axs[0, 2].set_xlim(0, signal_xlim_max)
axs[0, 2].text(0.5, 0.93, "Spike Scaling Factor = 2 (100 ms)", horizontalalignment='center', verticalalignment='center',
               transform=axs[0, 2].transAxes)
axs[0, 3].plot(time_axis, spike_signal_c_1, linewidth=1)
axs[0, 3].plot(time_axis, spike_signal_c_2, color="orange", linewidth=1)
axs[0, 3].axhline(y=0, color="black", linewidth=1.5)
axs[0, 3].set_ylim(signal_ylim_min, signal_ylim_max)
axs[0, 3].set_xlim(0, signal_xlim_max)
axs[0, 3].text(0.5, 0.93, "Spike Scaling Factor = 0 (100 ms)", horizontalalignment='center', verticalalignment='center',
               transform=axs[0, 3].transAxes)

# Plot full signals
axs[1, 0].plot(time_axis, spike_signal_a_1, linewidth=1)
axs[1, 0].plot(time_axis, spike_signal_a_2, color="orange", linewidth=1)
axs[1, 0].axhline(y=0, color="black", linewidth=1.5)
axs[1, 0].set_ylim(signal_ylim_min, signal_ylim_max)
axs[1, 0].set_xlim(0, signal_xlim_max)
axs[1, 1].plot(time_axis, signal_a_1, linewidth=1)
axs[1, 1].plot(time_axis, signal_a_2, color="orange", linewidth=1)
axs[1, 1].axhline(y=0, color="black", linewidth=1.5)
axs[1, 1].set_ylim(signal_ylim_min, signal_ylim_max)
axs[1, 1].set_xlim(0, signal_xlim_max)
axs[1, 2].plot(time_axis, signal_b_1, linewidth=1)
axs[1, 2].plot(time_axis, signal_b_2, color="orange", linewidth=1)
axs[1, 2].axhline(y=0, color="black", linewidth=1.5)
axs[1, 2].set_ylim(signal_ylim_min, signal_ylim_max)
axs[1, 2].set_xlim(0, signal_xlim_max)
axs[1, 3].plot(time_axis, signal_c_1, linewidth=1)
axs[1, 3].plot(time_axis, signal_c_2, color="orange", linewidth=1)
axs[1, 3].axhline(y=0, color="black", linewidth=1.5)
axs[1, 3].set_ylim(signal_ylim_min, signal_ylim_max)
axs[1, 3].set_xlim(0, signal_xlim_max)

# Plot the cross-correlation
time_axis_corr = np.linspace(-5 * 20, 5 * 20, len(raw_cross_corr_a))
axs[2, 0].plot(time_axis_corr, raw_cross_corr_a, color="black")
axs[2, 0].set_xlim(-5, 5)
axs[2, 0].set_ylim(-1e7, 2e7)
# Draw vertical line in max position
axs[2, 1].plot(time_axis_corr, raw_cross_corr_b, color="black")
axs[2, 1].set_xlim(-5, 5)
axs[2, 1].set_ylim(-1e7, 2e7)
axs[2, 2].plot(time_axis_corr, raw_cross_corr_c, color="black")
axs[2, 2].set_xlim(-5, 5)
axs[2, 2].set_ylim(-1e7, 2e7)
axs[2, 3].plot(time_axis_corr, raw_cross_corr_d, color="black")
axs[2, 3].set_xlim(-5, 5)
axs[2, 3].set_ylim(-1e7, 2e7)

# Remove y-axis labels from the second, third, and fourth column
for ax in axs[:, 1:].flatten():
    ax.set_yticklabels([])
    ax.set_ylabel("")
    ax.set_yticks([])

plt.tight_layout()
plt.show()


# Define the range of length multipliers
length_multipliers = [10, 50, 100, 200]

# Loop through each length multiplier
for multiplier in length_multipliers:
    # Create signal
    time_axis, signal_1, signal_2, spike_signal_1, spike_signal_2 = signal_creator(2, multiplier)

    # Compute the cross-correlation
    raw_cross_corr = signal.correlate(signal_2, signal_1, mode='full', method='fft')

    # Plot the cross-correlation
    title = str(int(multiplier / 10)) + "k"
    time_axis_corr = np.linspace(-5 * 20 * multiplier, 5 * 20 * multiplier, len(raw_cross_corr))
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis_corr, raw_cross_corr, color="black")
    plt.xlim(-5, 5)
    plt.ylim(-3e7, 9e7)
    plt.title(f"Spike Scaling Factor = 2 ({title} ms)")
    plt.show()


# Create signal
time_axis, signal_1, signal_2, spike_signal_1, spike_signal_2 = signal_creator(2, 100)

# Plot the signals
plt.figure(figsize=(10, 5))
plt.plot(time_axis, spike_signal_1, color="blue", linewidth=1)
plt.plot(time_axis, spike_signal_2, color="orange", linewidth=1)
plt.axhline(y=0, color="black", linewidth=1.5, linestyle="--")
plt.ylim(-70, 30)
plt.xlim(0, np.max(time_axis))
plt.title("Spike Scaling Factor = 2 (10k ms)")
plt.ylabel("Amplitude (µV)")
plt.xlabel("Time (ms)")
plt.show()

# Plot the signal with noise
plt.figure(figsize=(10, 5))
plt.plot(time_axis, signal_1, color="blue", linewidth=1)
plt.plot(time_axis, signal_2, color="orange", linewidth=1)
plt.axhline(y=0, color="black", linewidth=1.5, linestyle="--")
plt.ylim(-70, 30)
plt.xlim(0, np.max(time_axis))
plt.ylabel("Amplitude (µV)")
plt.xlabel("Time (ms)")
plt.title("Spike Scaling Factor = 2 (10k ms)")
plt.show()

# Define the range of length multipliers (from 5 to 300 in steps of 5)
length_multipliers = np.arange(1, 301, 1)
peak_values_dict = {}

# Define multiple value
mult_for_loop_list = [1.5, 2, 2.5, 3]

for mult_for_loop in mult_for_loop_list:
    # Reinitialize peak_values as an empty list for each multiple value
    peak_values = []

    # Loop through each length multiplier
    for length_multiplier in length_multipliers:
        # Generate signals based on the current length multiplier
        time_axis, signal_1, signal_2, spike_signal_1, spike_signal_2 = signal_creator(mult_for_loop, length_multiplier)

        # Compute cross-correlation between signal_2 and signal_1
        raw_cross_corr = signal.correlate(signal_2, signal_1, mode='full', method='fft')

        # Create a time axis for the cross-correlation result
        time_axis_corr = np.linspace(-5 * 20 * length_multiplier, 5 * 20 * length_multiplier, len(raw_cross_corr))

        # Filter the cross-correlation to keep values within the window [-5, 5] and set others to zero
        filtered_cross_corr = np.where(
            (time_axis_corr >= 1.9) & (time_axis_corr <= 2.1),
            raw_cross_corr,
            0
        )

        # Create a new array without zeros
        non_zero_cross_corr = filtered_cross_corr[filtered_cross_corr != 0]

        # Identify the peak value and its corresponding lag within the filtered window
        peak_pos = np.argmax(non_zero_cross_corr)
        peak_value = non_zero_cross_corr[peak_pos]
        peak_values.append(peak_value)

    # Add the peak values to the dictionary
    peak_values_dict[mult_for_loop] = peak_values

# Convert the peak values dict to a dataframe
df = pd.DataFrame(peak_values_dict)

# Walk through each column and check if value is 0, if so, set all previous values to 0 as well
for column in df.columns:
    for i in range(1, len(df[column])):
        if df[column].iloc[i] == 0:
            df[column].iloc[:i] = 0

# Extract columns from the dataframe
peak_values_1_5 = df[1.5].values
peak_values_2 = df[2].values
peak_values_2_5 = df[2.5].values
peak_values_3 = df[3].values

# Plot the peak values as line plot with length multiplier on x-axis as seconds (10 = 1 second)
# and one line for each Scaling Factor value
plt.figure(figsize=(10, 5))
plt.plot(peak_values_1_5, label="Scaling Factor = 1.5", markersize=3)
plt.plot(peak_values_2, label="Scaling Factor = 2")
plt.plot(peak_values_2_5, label="Scaling Factor = 2.5")
plt.plot(peak_values_3, label="Scaling Factor = 3")
plt.xlabel("Length multiplier [s]")
plt.ylabel("Peak value")
# Set x ticks to current value / 10
plt.xticks(ticks=np.arange(0, 301, 50), labels=(np.arange(0, 301, 50) / 10).astype(int))
###plt.yscale("log")
# Add a horizontal line at 0.40e8
plt.axhline(y=0.10e8 * 4, color="black", linestyle="--", linewidth=1)

plt.legend()
plt.title("Peak value of cross-correlation")
plt.show()
