
# Import libraries
import numpy as np
import pandas as pd
from scipy import signal


class SyntheticPropagatingNerveSignal():
    """
    This class generates synthetic propagating nerve signals
    with two traces. The first trace contains spikes that
    propagate to the second trace with a delay. The signal
    is stored as an analog signal and downsampled to a
    digital signal. Key metrics per spike are stored in a
    pandas dataframe.
    """
    # Class variables
    analog_sampling_rate = 3000000  # 3 MHz
    electrode_distance = 0.002  # 0.002 m = 2 mm

    # Define signal dataframe columns
    signal_columns = ['Scaling Factor', 'True Velocity', 'Analog Peak 1', 'Analog Peak 2', 'Digital Peak 1', 'Digital Peak 2']

    # Constructor
    def __init__(self, spike_velocities=[1, 5, 10, 30, 60, 120], spike_probability=0.00005, duration_s=1,
                 high_amp_mult=12, low_amp_mult=6, noise_mult=6, digital_sampling_rate=30000):
        self.duration_s = duration_s
        self.spike_probability = spike_probability
        self.high_amp_mult = high_amp_mult
        self.low_amp_mult = low_amp_mult
        self.noise_mult = noise_mult
        self.spike_velocities = spike_velocities
        self.digital_sampling_rate = digital_sampling_rate
        self.downsampling_factor = self.analog_sampling_rate / self.digital_sampling_rate
        self.signal_df = pd.DataFrame(columns=self.signal_columns)
        self.a_time_axis = np.linspace(0, duration_s, duration_s * self.analog_sampling_rate)
        self.a_time_axis_ms = np.linspace(0, duration_s * 1000, duration_s * self.analog_sampling_rate)
        self.d_time_axis = np.linspace(0, duration_s, duration_s * self.digital_sampling_rate)
        self.d_time_axis_ms = np.linspace(0, duration_s * 1000, duration_s * self.digital_sampling_rate)
        self.a_spike_trace_1, self.a_spike_trace_2 = self.create_spike_traces(self.a_time_axis)
        self.d_spike_trace_1 = self.a_spike_trace_1[::round(self.downsampling_factor)]
        self.d_spike_trace_2 = self.a_spike_trace_2[::round(self.downsampling_factor)]
        self.a_noise_signal_1 = WhiteNoise(duration_s, noise_mult).noise_signal
        self.a_noise_signal_2 = WhiteNoise(duration_s, noise_mult).noise_signal
        self.a_combined_signal_1 = self.a_spike_trace_1 + self.a_noise_signal_1
        self.a_combined_signal_2 = self.a_spike_trace_2 + self.a_noise_signal_2
        self.d_combined_signal_1 = self.a_combined_signal_1[::round(self.downsampling_factor)]
        self.d_combined_signal_2 = self.a_combined_signal_2[::round(self.downsampling_factor)]

    # Define function to generate spike traces
    def create_spike_traces(self, time_axis, electrode_distance=electrode_distance):
        # Initialize spike signals
        spike_signal_1 = np.zeros(len(self.a_time_axis))
        spike_signal_2 = np.zeros(len(self.a_time_axis))

        # Loop through time axis and generate spikes
        sample_counter = 30000
        while sample_counter < len(self.a_time_axis):
            # Break if within last 30000 samples
            if sample_counter > len(self.a_time_axis) - 30000:
                break

            # Decide if a spike should be generated
            if np.random.rand() < self.spike_probability:
                # Randomly choose spike amp, speed, and direction
                mult = np.random.choice([self.high_amp_mult, self.low_amp_mult])
                speed = np.random.choice(self.spike_velocities)
                direction = np.random.choice([-1, 1])
                velocity = direction * speed
                spike = SingleExtraCellularSpike(mult).spike_signal

                # Add the spike to the first trace
                spike_signal_1[sample_counter:sample_counter + len(spike)] = spike

                # Calculate the time it takes for the spike to travel 2 mm
                spike_time = electrode_distance / velocity
                spike_sample_diff = round(spike_time * self.analog_sampling_rate)

                # Add the spike to the second trace
                spike_signal_2[
                sample_counter + spike_sample_diff:sample_counter + spike_sample_diff + len(spike)] = spike

                # Add the spike to the signal dataframe
                analog_peak1 = sample_counter + SingleExtraCellularSpike(mult).max_pos
                analog_peak2 = sample_counter + SingleExtraCellularSpike(mult).max_pos + spike_sample_diff
                digital_peak1 = round(analog_peak1 / self.downsampling_factor)
                digital_peak2 = round(analog_peak2 / self.downsampling_factor)

                self.signal_df.loc[len(self.signal_df)] = [mult, velocity, analog_peak1, analog_peak2,
                                                           digital_peak1, digital_peak2]

                # Increment sample counter
                sample_counter += 30000
            else:
                # Increment sample counter
                sample_counter += 1

        # Return the spike signals
        return spike_signal_1, spike_signal_2


class SingleExtraCellularSpike():
    """
    This class generates a single extracellular spike signal. The spike signal
    is stored as an analog signal and can be used to generate
    synthetic propagating nerve signals.
    """
    # Class variables
    analog_sampling_rate = 3000000  # 1 MHz
    time_axis = np.linspace(0, 10, round(0.005 * analog_sampling_rate))  # 0.005 seconds = 5 ms

    # Constructor
    def __init__(self, spike_mult):
        self.spike_mult = spike_mult
        self.spike_signal = self.create_spike_signal(self.spike_mult)
        self.max_pos = np.argmin(self.spike_signal)

    # Generate spike based on multiple value
    def create_spike_signal(self, spike_mult, time_axis=time_axis):
        # Generate spike signal
        spike_signal = 10 * (-1.8 * np.exp(-3 * (time_axis - 3.5) ** 2) +
                             0.3 * np.exp(-0.4 * (time_axis - 5.3) ** 2) +
                             0.1 * np.exp(-0.4 * (time_axis - 5) ** 2))

        # Make first and last samples of the signal zero
        spike_signal[:1] = 0
        spike_signal[-1:] = 0

        # Normalize and scale the spike
        spike_signal = spike_signal / np.sqrt(np.mean(spike_signal ** 2))
        spike_signal = spike_signal * spike_mult

        # Return the spike signal
        return spike_signal


class WhiteNoise():
    """
    This class generates white noise signal with a given.
    The noise signal is stored
    as an analog signal and can be used to generate synthetic
    propagating nerve signals.
    """
    # Class variables
    analog_sampling_rate = 3000000  # 1 MHz

    # Constructor
    def __init__(self, duration_s, noise_mult):
        self.duration_s = duration_s
        self.noise_mult = noise_mult
        self.time_axis = np.linspace(0, duration_s, duration_s * self.analog_sampling_rate)
        self.noise_signal = self.generate_noise(self.noise_mult, self.time_axis)

    # Generate white noise signal
    def generate_noise(self, noise_mult, time_axis):
        # Generate noise signal
        white_noise = np.random.normal(0, noise_mult, len(time_axis))
        noise = LowPassFilter(1000).filter_signal(white_noise)

        # Normalize and scale the noise
        noise = noise / np.sqrt(np.mean(noise ** 2))
        noise = noise * noise_mult

        # Return the noise signal
        return noise


class LowPassFilter():
    """
    This class generates a low pass filter with a given
    cutoff frequency. The filter can be used to filter
    the noise for the synthetic propagating nerve signals.
    """
    # Class variables
    analog_sampling_rate = 3000000  # 1 MHz

    # Constructor
    def __init__(self, cutoff_freq):
        self.cutoff_freq = cutoff_freq
        self.nyquist = 0.5 * self.analog_sampling_rate
        self.normal_cutoff = self.cutoff_freq / self.nyquist
        self.b, self.a = signal.butter(5, self.normal_cutoff, btype='low', analog=False)

    # Filter the input signal
    def filter_signal(self, input_signal):
        return signal.lfilter(self.b, self.a, input_signal)
