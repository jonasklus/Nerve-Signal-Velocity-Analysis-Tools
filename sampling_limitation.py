
# Import libraries
import numpy as np
import matplotlib.pyplot as plt


# Define the Gaussian function
def gaussian(x, mu=0, sigma=0.1):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) * 100


# Define function for sinc interpolation
def sinc_interpolation_fft(digital_signal, upsampling_factor):
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


# Define color
color1 = "orange"
color2 = "blue"

# Start figure
fig, axs = plt.subplots(1, 3, figsize=(20, 7))

# Define the first subplot
ax = axs[0]

# Create a Gaussian curve
x = np.linspace(-1, 1, 1000)
y = gaussian(x)

# Record every 10th value but start at the 5th value
offset = 0
x_sampled = x[offset::20]
y_sampled = y[offset::20]

# Run sinc interpolation
y_upsampled = sinc_interpolation_fft(y_sampled, 10)
sinc_offset = (offset * 2 - 1) / 1000
x_upsampled = np.linspace(-1 + sinc_offset, 1 + sinc_offset, len(y_upsampled))

# Plot the Gaussian curve
ax.plot(x, y, label='Analog Signal', color="darkgrey", linestyle="--")  # Gaussian function in black

# Scatter plot for the markers
ax.scatter(x_sampled, y_sampled, s=50, label='Digital Sampling Points', zorder=5, color=color1)
ax.set_xlim(-0.1, 0.1)
ax.set_ylim(90, 102)

# Draw horizontal line for max marker and label it with text within the raw with Max: value
ax.axhline(y=max(y_sampled), linestyle='--', label='Max Marker', color=color1)
ax.text(-0.07, max(y_sampled)+0.2, f'Max: {max(y_sampled):.0f}', color=color1, verticalalignment='bottom', horizontalalignment='right')

# Get rid if xticks
ax.set_xticks([])


# Define the second subplot
ax = axs[1]

# Create a Gaussian curve
x = np.linspace(-1, 1, 1000)
y = gaussian(x)

# Record every 10th value but start at the 5th value
offset = 10
x_sampled = x[offset::20]
y_sampled = y[offset::20]

# Run sinc interpolation
y_upsampled = sinc_interpolation_fft(y_sampled, 10)
sinc_offset = (offset * 2 - 1) / 1000
x_upsampled = np.linspace(-1 + sinc_offset, 1 + sinc_offset, len(y_upsampled))

# Plot the Gaussian curve
ax.plot(x, y, label='Analog Signal', color="darkgrey", linestyle="--")  # Gaussian function in black

# Scatter plot for the markers
ax.scatter(x_sampled, y_sampled, s=50, label='Digital Sampling Points', zorder=5, color=color1)
ax.set_xlim(-0.1, 0.1)
ax.set_ylim(90, 102)

# Draw horizontal line for max marker and label it with text within the raw with Max: value
ax.axhline(y=max(y_sampled), linestyle='--', label='Max Marker', color=color1)
ax.text(-0.072, max(y_sampled)+0.2, f'Max: {max(y_sampled):.0f}', color=color1, verticalalignment='bottom', horizontalalignment='right')

# Get rid if xticks
ax.set_xticks([])
ax.set_yticks([])


# Define the third subplot
ax = axs[2]

# Create a Gaussian curve
x = np.linspace(-1, 1, 1000)
y = gaussian(x)

# Record every 10th value but start at the 5th value
offset = 10
x_sampled = x[offset::20]
y_sampled = y[offset::20]

# Run sinc interpolation
y_upsampled = sinc_interpolation_fft(y_sampled, 10)
sinc_offset = (offset * 2 - 1) / 1000
x_upsampled = np.linspace(-1 + sinc_offset, 1 + sinc_offset, len(y_upsampled))

# Plot the Gaussian curve
ax.plot(x, y, label='Analog Signal', color="darkgrey", linestyle="--")  # Gaussian function in black

# Scatter plot for the markers
ax.scatter(x_sampled, y_sampled, s=50, label='Digital Sampling Points', zorder=5, color=color1)
ax.set_xlim(-0.1, 0.1)
ax.set_ylim(90, 102)

# Get rid if xticks
ax.set_xticks([])
ax.set_yticks([])

# Plot the upsampled signal
ax.plot(x_upsampled, y_upsampled, "D", label="Sinc Interpolation Points", color=color2, markersize=4)

# Draw horizontal line for max marker and label it with text within the raw with Max: value
ax.axhline(y=max(y_upsampled), linestyle='--', color=color2)
ax.text(-0.07, max(y_upsampled)+0.2, f'Max: {max(y_upsampled):.0f}', color=color2, verticalalignment='bottom', horizontalalignment='right')
ax.legend()

# Show the plot
plt.subplots_adjust(wspace=0.05)
plt.show()
