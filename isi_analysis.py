
# Import libraries
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import CubicSpline


class InterSpikeIntervallAnalyzer:
    """
    This class analyzes the interspike interval of a propagating nerve signal
    and compares the results of different interpolation methods as well as
    the correlation velocity method.
    """
    # Constructor
    def __init__(self, SyntheticPropagatingNerveSignal, sd_threshold, ahc_d_sample_window=100):
        self.nerve_signal = SyntheticPropagatingNerveSignal
        self.sd_threshold = sd_threshold
        self.ahc_d_sample_window = ahc_d_sample_window
        self.upsampling_factor = round(self.nerve_signal.analog_sampling_rate / self.nerve_signal.digital_sampling_rate)
        self.sinc_s_sample_window = self.ahc_d_sample_window * self.upsampling_factor
        self.nerve_signal.s_combined_signal_1 = self.sinc_interpolation_fft(self.nerve_signal.d_combined_signal_1, self.upsampling_factor)
        self.nerve_signal.s_combined_signal_2 = self.sinc_interpolation_fft(self.nerve_signal.d_combined_signal_2, self.upsampling_factor)
        self.nerve_signal.cs_combined_signal_1 = self.cubic_spline_interpolation(self.nerve_signal.d_combined_signal_1, self.upsampling_factor)
        self.nerve_signal.cs_combined_signal_2 = self.cubic_spline_interpolation(self.nerve_signal.d_combined_signal_2, self.upsampling_factor)


        # Run alternating hill climber on digital signal
        self.alternating_hill_climber(self.nerve_signal, self.nerve_signal.d_combined_signal_1, self.nerve_signal.d_combined_signal_2,
                                      self.sd_threshold, self.ahc_d_sample_window, 'AHC', 1)

        # Run alternating hill climber on sinc interpolated signal
        self.alternating_hill_climber(self.nerve_signal, self.nerve_signal.s_combined_signal_1, self.nerve_signal.s_combined_signal_2,
                                      self.sd_threshold, self.sinc_s_sample_window, 'SINC', self.upsampling_factor)

        # Run alternating hill climber on cubic spline interpolated signal
        self.alternating_hill_climber(self.nerve_signal, self.nerve_signal.cs_combined_signal_1, self.nerve_signal.cs_combined_signal_2,
                                      self.sd_threshold, self.sinc_s_sample_window, 'SPLINE', self.upsampling_factor)

        # Run correlation interspike interval
        self.correlation_isi(self.nerve_signal, self.nerve_signal.d_combined_signal_1, self.nerve_signal.d_combined_signal_2)



    # Define function for alternating hill climber
    def alternating_hill_climber(self, nerve_signal, signal_1, signal_2, sd_threshold, sample_window, method, df_search_scaler):
        # Add columns to the signal dataframe
        nerve_signal.signal_df[f'{method} Actual Spike Detected'] = np.nan
        nerve_signal.signal_df[f'{method} Detected Peak 1'] = np.nan
        nerve_signal.signal_df[f'{method} Detected Peak 2'] = np.nan
        nerve_signal.signal_df[f'{method} Detected Velocity'] = np.nan
        nerve_signal.signal_df[f'{method} Correct Velocity Detected'] = np.nan

        # Copy signal dataframe as false_positive_df
        nerve_signal.false_positive_df = nerve_signal.signal_df.copy()
        nerve_signal.false_positive_df = nerve_signal.false_positive_df[0:0]

        # Calculate amplitude corresponding to the threshold
        sd_amp = np.std(signal_1)
        sd_amp_threshold = np.std(signal_1) * sd_threshold

        # Initialize variables
        curr_peak_signal_1 = np.nan
        curr_peak_signal_2 = np.nan
        curr_velocity = np.nan

        # Create shifted versions of signal_1 for comparison
        shifted_left = np.roll(signal_1, 1)
        shifted_right = np.roll(signal_1, -1)

        # Create a boolean array where the peaks are located
        peaks_mask = (signal_1 < -sd_amp_threshold) & (signal_1 < shifted_left) & (signal_1 < shifted_right)

        # Use np.where to get the indices of these peaks
        peak_indices = np.where(peaks_mask)[0]

        # Remove peaks that are too close to the beginning or end of the signal (30k)
        peak_indices = peak_indices[(peak_indices >= 100) & (peak_indices <= len(signal_1) - 100)]

        for i in peak_indices:
            curr_peak_signal_1 = i
            curr_peak_signal_2 = np.nan
            # Crawl through signal 2 to find the first spike above the threshold around the peak in signal 1
            for j in range(sample_window):
                # Check if j is the first or last index
                if i + j <= 0 or i + j >= len(signal_2) - 1 or i - j <= 0 or i - j >= len(signal_2) - 1:
                    continue
                # First check if there is a peak in the positive direction i + j
                if (signal_2[ i +j] < -sd_amp_threshold and
                        signal_2[ i + j] < signal_2[ i + j - 1] and
                        signal_2[ i + j] < signal_2[ i + j + 1]):
                    curr_peak_signal_2 = i + j
                    break
                # Then check if there is a peak in the negative direction i - j
                elif (signal_2[i - j] < -sd_amp_threshold and
                      signal_2[i - j] < signal_2[i - j - 1] and
                      signal_2[i - j] < signal_2[i - j + 1]):
                    curr_peak_signal_2 = i - j
                    break

            # Calculate the velocity of the detected peak if both peaks are found
            curr_velocity = np.nan
            if np.isnan(curr_peak_signal_1) == False and np.isnan(curr_peak_signal_2) == False:
                if curr_peak_signal_1 == curr_peak_signal_2:
                    curr_velocity = np.inf
                else:
                    curr_velocity = ((nerve_signal.electrode_distance * nerve_signal.digital_sampling_rate * df_search_scaler) /
                                     (curr_peak_signal_2 - curr_peak_signal_1))

            # Loop through the signal dataframe and update the values if curr_peak_signal_1 matches the Digital Peak 1 within sample_window and curr_peak_signal_2 matches the Digital Peak 2 within sample_window
            found_in_df = False
            for k in range(len(nerve_signal.signal_df)):
                # Continue if True Velocity is NaN
                if np.isnan(nerve_signal.signal_df.loc[k, 'True Velocity']):
                    continue

                # Check if the peak 1 is within the window of the current peak and update the signal dataframe
                if (round(nerve_signal.signal_df.loc[k, 'Digital Peak 1'] * df_search_scaler) > curr_peak_signal_1 - sample_window and
                    round(nerve_signal.signal_df.loc[k, 'Digital Peak 1'] * df_search_scaler) < curr_peak_signal_1 + sample_window):
                    nerve_signal.signal_df.loc[k, f'{method} Actual Spike Detected'] = True
                    nerve_signal.signal_df.loc[k, f'{method} Detected Peak 1'] = curr_peak_signal_1
                    nerve_signal.signal_df.loc[k, f'{method} Detected Peak 2'] = curr_peak_signal_2
                    nerve_signal.signal_df.loc[k, f'{method} Detected Velocity'] = curr_velocity

                    # Set velocity to false if curr_velocity is not NaN
                    if np.isnan(curr_velocity) == False:
                        nerve_signal.signal_df.loc[k, f'{method} Correct Velocity Detected'] = False

                    # If the velocity is within 40% of the actual velocity and has the same sign, the velocity is correct
                    if (abs(curr_velocity) <= 1.4 * abs(nerve_signal.signal_df.loc[k, 'True Velocity']) and
                            abs(curr_velocity) >= 0.6 * abs(nerve_signal.signal_df.loc[k, 'True Velocity']) and
                            np.sign(curr_velocity) == np.sign(nerve_signal.signal_df.loc[k, 'True Velocity'])):
                        nerve_signal.signal_df.loc[k, f'{method} Correct Velocity Detected'] = True

                    # Break the loop if the peak is found
                    found_in_df = True
                    break

            # If there is a peak 1 and peak 2 but the peak is not found in the signal dataframe, add it to the false positive dataframe
            if found_in_df == False and np.isnan(curr_peak_signal_1) == False and np.isnan(curr_peak_signal_2) == False:
                if method == 'AHC':
                    # Add the row
                    nerve_signal.false_positive_df.loc[len(nerve_signal.false_positive_df)] = [np.nan, np.nan, np.nan,
                                                                                               np.nan, np.nan, np.nan,
                                                                                               False, curr_peak_signal_1,
                                                                                               curr_peak_signal_2,
                                                                                               curr_velocity, np.nan]
                elif method == 'SINC':
                    # Check if the peak is close to the peak of a false positive already in the signal df as AHC detected peak
                    for k in range(len(nerve_signal.signal_df)):
                        if (np.isnan(nerve_signal.signal_df.loc[k, 'AHC Detected Peak 1']) == False and
                            round(nerve_signal.signal_df.loc[k, 'AHC Detected Peak 1'] * df_search_scaler) > curr_peak_signal_1 - sample_window and
                            round(nerve_signal.signal_df.loc[k, 'AHC Detected Peak 1'] * df_search_scaler) < curr_peak_signal_1 + sample_window and
                            nerve_signal.signal_df.loc[k, 'AHC Actual Spike Detected'] == False):
                            # Fill the values of the false positive for the SINC fields
                            nerve_signal.signal_df.loc[k, 'SINC Actual Spike Detected'] = False
                            nerve_signal.signal_df.loc[k, 'SINC Detected Peak 1'] = curr_peak_signal_1
                            nerve_signal.signal_df.loc[k, 'SINC Detected Peak 2'] = curr_peak_signal_2
                            nerve_signal.signal_df.loc[k, 'SINC Detected Velocity'] = curr_velocity
                            break
                    else:
                        # Add a new row to the false positive dataframe
                        nerve_signal.false_positive_df.loc[len(nerve_signal.false_positive_df)] = [np.nan, np.nan, np.nan,
                                                                                                   np.nan, np.nan, np.nan,
                                                                                                   np.nan, np.nan, np.nan,
                                                                                                   np.nan, np.nan,
                                                                                                   False, curr_peak_signal_1,
                                                                                                   curr_peak_signal_2,
                                                                                                   curr_velocity, np.nan]

                elif method == 'SPLINE':
                    # Check if the peak is close to the peak of a false positive already in the signal df as AHC detected peak
                    for k in range(len(nerve_signal.signal_df)):
                        if (np.isnan(nerve_signal.signal_df.loc[k, 'AHC Detected Peak 1']) == False and
                            round(nerve_signal.signal_df.loc[k, 'AHC Detected Peak 1'] * df_search_scaler) > curr_peak_signal_1 - sample_window and
                            round(nerve_signal.signal_df.loc[k, 'AHC Detected Peak 1'] * df_search_scaler) < curr_peak_signal_1 + sample_window and
                            nerve_signal.signal_df.loc[k, 'AHC Actual Spike Detected'] == False):
                            # Fill the values of the false positive for the SINC fields
                            nerve_signal.signal_df.loc[k, 'SPLINE Actual Spike Detected'] = False
                            nerve_signal.signal_df.loc[k, 'SPLINE Detected Peak 1'] = curr_peak_signal_1
                            nerve_signal.signal_df.loc[k, 'SPLINE Detected Peak 2'] = curr_peak_signal_2
                            nerve_signal.signal_df.loc[k, 'SPLINE Detected Velocity'] = curr_velocity
                            break
                    else:
                        # Add a new row to the false positive dataframe
                        nerve_signal.false_positive_df.loc[len(nerve_signal.false_positive_df)] = [np.nan, np.nan, np.nan,
                                                                                                   np.nan, np.nan, np.nan,
                                                                                                   np.nan, np.nan, np.nan,
                                                                                                   np.nan, np.nan, np.nan,
                                                                                                   np.nan, np.nan, np.nan,
                                                                                                   np.nan,
                                                                                                   False, curr_peak_signal_1,
                                                                                                   curr_peak_signal_2,
                                                                                                   curr_velocity, np.nan]

        # Add the false positive dataframe to the signal dataframe
        nerve_signal.signal_df = pd.concat([nerve_signal.signal_df, nerve_signal.false_positive_df], axis=0,
                                           ignore_index=True)


    # Define function for sinc interpolation
    def sinc_interpolation_fft(self, digital_signal, upsampling_factor):
        """
        Fast Fourier Transform (FFT) based sinc or bandlimited interpolation.

        Args:
            x (np.ndarray): signal to be interpolated, can be 1D or 2D
            s (np.ndarray): time points of x (*s* for *samples*)
            u (np.ndarray): time points of y (*u* for *upsampled*)

        Returns:
            np.ndarray: interpolated signal at time points *u*
        """
        # Define parameters
        num_output = len(digital_signal) * upsampling_factor
        x = digital_signal
        s = digital_signal.shape[0]

        # Compute the FFT of the input signal
        X = np.fft.rfft(x)

        # Create a new array for the zero-padded frequency spectrum
        X_padded = np.zeros(num_output // 2 + 1, dtype=complex)

        # Copy the original frequency spectrum into the zero-padded array
        X_padded[:X.shape[0]] = X

        # Compute the inverse FFT of the zero-padded frequency spectrum
        x_interpolated = np.fft.irfft(X_padded, n=num_output)

        # Create output signal
        output_signal_fft = x_interpolated * (num_output / s)

        # Return the interpolated signal
        return output_signal_fft


    # Define function for cubic spline interpolation
    def cubic_spline_interpolation(self, digital_signal, upsampling_factor):
        # Number of samples in the original signal along the interpolation axis
        num_original = digital_signal.shape[0]

        # Original time points (assuming unit sampling interval)
        t_original = np.arange(num_original)

        # Number of samples in the interpolated signal
        num_output = num_original * upsampling_factor

        # New time points for the interpolated signal
        t_new = np.linspace(t_original[0], t_original[-1], num=num_output)

        # Perform cubic spline interpolation along the first axis
        cs = CubicSpline(t_original, digital_signal, axis=0)

        # Evaluate the interpolator at the new time points
        output_signal = cs(t_new)

        return output_signal


    # Define function for correlation velocity estimation
    def correlation_isi(self, nerve_signal, signal_1, signal_2):
        # Create columns in the signal dataframe
        nerve_signal.signal_df[f'CORR Detected Velocity'] = np.nan
        nerve_signal.signal_df[f'CORR Correct Velocity Detected'] = np.nan

        # Loop through each row with iterrows
        for index, row in nerve_signal.signal_df.iterrows():
            # Initialize variables
            velocity_corr = np.nan

            # Check if AHC Detected Peak 1 and AHC Detected Peak 2 are not NaN
            if not np.isnan(row["AHC Detected Peak 1"]) and not np.isnan(row["AHC Detected Peak 2"]):
                # Run a cross-correlation between the two signals 100 samples before and after the detected peaks in both directions
                start = int(row["AHC Detected Peak 1"]) - 100
                end = int(row["AHC Detected Peak 1"]) + 100
                s1 = signal_1[start:end]
                s2 = signal_2[start:end]

                corr_true = signal.correlate(s2, s1, "full")

                # Find the maximum correlation value and its index
                max_pos_true = np.argmax(corr_true)

                # Find the maximum cross-correlation value and calculate the velocity
                sig_delay_samples = max_pos_true - 199

                velocity_corr = ((nerve_signal.electrode_distance * nerve_signal.digital_sampling_rate) /
                                      sig_delay_samples)

                # Assign the calculated velocity to the spike dataframe
                nerve_signal.signal_df.at[index, "CORR Detected Velocity"] = velocity_corr

            # Set velocity to false if velocity_corr is not NaN and True Velocity is not NaN
            if np.isnan(velocity_corr) == False and np.isnan(row["True Velocity"]) == False:
                nerve_signal.signal_df.at[index, "CORR Correct Velocity Detected"] = False

            # If the velocity is within 40% of the actual velocity and has the same sign, the velocity is correct
            if (abs(velocity_corr) <= 1.4 * abs(row["True Velocity"]) and
                abs(velocity_corr) >= 0.6 * abs(row["True Velocity"]) and
                np.sign(velocity_corr) == np.sign(row["True Velocity"])):
                nerve_signal.signal_df.at[index, "CORR Correct Velocity Detected"] = True
