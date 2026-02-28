import numpy as np
import matplotlib.pyplot as plt

def plot_magnitude_spectrum(frequency_domain: np.array, plot_radial: bool = True):
    """
    Computes and plots the Magnitude Spectrum of a frequency domain image.
    
    Includes an optional radial energy plot to show how much information 
    is contained at different frequency distances.
    
    Args:
        frequency_domain: 2D numpy array of complex numbers (DFT output).
        plot_radial: Boolean to enable/disable the radial energy plot.
    """
    
    # --- Step 1: Shift zero-frequency to center ---
    # This moves the DC component (average brightness) from the top-left corner
    # to the visual center of the image.
    f_shifted = np.fft.fftshift(frequency_domain)
    
    # --- Step 2: Compute Magnitude ---
    # The magnitude of a complex number a + bi is sqrt(a^2 + b^2).
    # numpy.abs() handles this automatically for complex arrays.
    magnitude = np.abs(f_shifted)
    
    # --- Step 3: Log-Scale Transformation ---
    # The dynamic range of Fourier coefficients is huge (e.g., center is 1,000,000,
    # high freq is 1). We use log(1 + magnitude) to compress this so we can 
    # actually see the details.
    magnitude_log = np.log(1 + magnitude)
    
    # --- Step 4: Setup Plotting ---
    if plot_radial:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
        
    # Plot the 2D Spectrum
    # 
    img_plot = ax1.imshow(magnitude_log, cmap='gray')
    ax1.set_title("Magnitude Spectrum (Log Scale)")
    ax1.axis('off') # Hide axes ticks for cleaner look
    fig.colorbar(img_plot, ax=ax1, fraction=0.046, pad=0.04)
    
    # --- Optional Extension: Radial Frequency Energy ---
    if plot_radial:
        # Get dimensions
        rows, cols = magnitude.shape
        center_row, center_col = rows // 2, cols // 2
        
        # Create coordinate grids (Y, X)
        y, x = np.ogrid[:rows, :cols]
        
        # Calculate distance of every pixel from the center
        distance_from_center = np.sqrt((y - center_row)**2 + (x - center_col)**2)
        
        # Convert distances to integers to use them as "bins"
        distance_int = distance_from_center.astype(int)
        
        # Sum the magnitude energy for each integer radius
        # np.bincount sums the values in 'weights' based on the index in 'x'
        radial_energy = np.bincount(distance_int.ravel(), weights=magnitude.ravel())
        
        # Plot the 1D Profile
        ax2.plot(radial_energy, color='blue', linewidth=1.5)
        ax2.set_title("Radial Energy Profile")
        ax2.set_xlabel("Frequency Radius (Distance from Center)")
        ax2.set_ylabel("Total Magnitude Energy")
        ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

# --- Integration Example ---
# Assuming you have the 'frequency_domain_array' from the previous script:
# plot_magnitude_spectrum(frequency_domain_array)

