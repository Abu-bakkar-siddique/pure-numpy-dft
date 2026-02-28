import numpy as np

def apply_low_pass_filter(frequency_domain: np.array, radius: float) -> np.array:
    """
    Applies an Ideal Low Pass Filter (ILPF) to a frequency domain array.
    
    This keeps frequencies inside a center circle (low freqs) and cuts 
    frequencies outside (high freqs), resulting in a blurred image.
    
    Args:
        frequency_domain: 2D numpy array of complex numbers (DFT output).
        radius: The cutoff distance. Smaller radius = more blur.
    
    Returns:
        Filtered 2D numpy array of complex numbers.
    """
    
    # 1. Get dimensions
    rows, cols = frequency_domain.shape
    center_row, center_col = rows // 2, cols // 2
    
    # 2. Shift the Zero-Frequency component to the center
    # Without this, the low frequencies are at the corners, and a 
    # center mask would delete the most important data.
    shifted_frequencies = np.fft.fftshift(frequency_domain)
    
    # 3. Create a coordinate grid (Vectorized, like before)
    # y represents row indices, x represents column indices
    y_indices = np.arange(rows).reshape((rows, 1)) # Vertical Column
    x_indices = np.arange(cols).reshape((1, cols)) # Horizontal Row
    
    # 4. Calculate Distance Matrix from the center (Using Pythagoras)
    # Broadcasting creates a 2D array of distances for every pixel
    distance_from_center = np.sqrt((y_indices - center_row)**2 + 
                                   (x_indices - center_col)**2)
    
    # 5. Create the Mask (Ideal Filter)
    # 1 if inside radius, 0 if outside
    mask = distance_from_center <= radius
    
    # 6. Apply the mask (Element-wise multiplication)
    filtered_shifted = shifted_frequencies * mask
    
    # 7. Shift back to original positions (Inverse Shift)
    filtered_output = np.fft.ifftshift(filtered_shifted)
    
    return filtered_output

# --- Example Usage (Pseudo-code context) ---
# frequency_array = discrete_fourier_transform(image)
# filtered_freq = apply_low_pass_filter(frequency_array, radius=30)
# final_image = inverse_discrete_fourier_transform(filtered_freq)


def apply_high_pass_filter(frequency_domain: np.array, radius: float) -> np.array:
    """
    Applies an Ideal High Pass Filter (IHPF) to a frequency domain array.
    
    This cuts frequencies inside a center circle (low freqs) and keeps 
    frequencies outside (high freqs), resulting in an image showing only edges.
    
    Args:
        frequency_domain: 2D numpy array of complex numbers (DFT output).
        radius: The cutoff distance. Larger radius = only finest edges remain.
    
    Returns:
        Filtered 2D numpy array of complex numbers.
    """
    
    # 1. Get dimensions
    rows, cols = frequency_domain.shape
    center_row, center_col = rows // 2, cols // 2
    
    # 2. Shift the Zero-Frequency component to the center
    # We need the low frequencies in the middle so we can "punch them out".
    shifted_frequencies = np.fft.fftshift(frequency_domain)
    
    # 3. Create a coordinate grid (Vectorized)
    y_indices = np.arange(rows).reshape((rows, 1))
    x_indices = np.arange(cols).reshape((1, cols))
    
    # 4. Calculate Distance Matrix from the center
    distance_from_center = np.sqrt((y_indices - center_row)**2 + 
                                   (x_indices - center_col)**2)
    
    # 5. Create the Mask (The "Donut")
    # 1 if OUTSIDE radius (Keep), 0 if INSIDE radius (Cut)
    # This removes the average brightness (DC component) and smooth areas.
    mask = distance_from_center > radius
    
    # 6. Apply the mask
    filtered_shifted = shifted_frequencies * mask
    
    # 7. Shift back to original positions
    filtered_output = np.fft.ifftshift(filtered_shifted)
    
    return filtered_output