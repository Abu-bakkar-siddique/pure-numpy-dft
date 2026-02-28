import numpy as np
from PIL import Image
from filters import apply_low_pass_filter
from plots import plot_magnitude_spectrum

def discrete_fourier_transform(grayscale_image: np.array) -> np.array: 
    """
    Convert a grayscale image from spatial domain to frequency domain using 
    vectorized matrix multiplication (DFT).
    
    Formula: F = U @ I @ V
    Where U is the vertical transform matrix, I is the image, and V is the horizontal transform matrix.
    
    Args:
        grayscale_image: A 2D numpy array representing grayscale pixel intensities.
        
    Returns:
        A 2D numpy array of complex numbers representing the frequency domain.
    """
    
    # Get dimensions of the input image
    height, width = grayscale_image.shape
    
    # --- Step 1: Create the Row-wise Transform Matrix (Width x Width) ---
    # We create a 2D array of indices (k, n) to compute exponents for the width
    # n: spatial index (0 to width-1), k: frequency index (0 to width-1)
    
    n_indices = np.arange(width) #arranged row wise
    
    print(f" n_indices : {n_indices}")
    k_indices = n_indices.reshape((width, 1)) #arranged column wise

    print(f" k_indices : {k_indices}")

    # Calculate the exponent matrix: -2πi * (k * n) / N
    # We use vector broadcasting here to create a 2D matrix of exponents
    exponent_cols = -2j * np.pi * k_indices * n_indices / width
    
    # The transform matrix for columns (W_width)
    col_transform_matrix = np.exp(exponent_cols)
    
    # --- Step 2: Create the Column-wise Transform Matrix (Height x Height) ---
    # m: spatial index (0 to height-1), l: frequency index (0 to height-1)
    
    m_indices = np.arange(height)
    l_indices = m_indices.reshape((height, 1))
    
    # Calculate the exponent matrix: -2πi * (l * m) / M
    exponent_rows = -2j * np.pi * l_indices * m_indices / height
    
    # The transform matrix for rows (W_height)
    dft_matrix_rows = np.exp(exponent_rows)
    
    # --- Step 3: Perform Matrix Multiplication ---
    # We apply the transform. Mathematically, 2D DFT is separable.
    # 1. dft_matrix_rows @ grayscale_image -> Transforms the columns
    # 2. Result @ col_transform_matrix -> Transforms the rows
 
    frequency_domain = dft_matrix_rows @ grayscale_image @ col_transform_matrix 
    
    return frequency_domain

def inverse_discrete_fourier_transform(frequency_domain: np.array) -> np.array:
    """
    Convert frequency domain representation back to spatial domain using IDFT.
    
    Args:
        frequency_domain: A 2D numpy array of complex numbers.
        
    Returns:
        A 2D numpy array of real numbers representing the reconstructed image.
    """
    
    # Get dimensions of the frequency data
    height, width = frequency_domain.shape
    
    # --- Step 1: Create the Row-wise Inverse Matrix ---
    # The logic is identical to DFT, but the exponent sign is POSITIVE
    n_indices = np.arange(width)
    k_indices = n_indices.reshape((width, 1))
    
    # Exponent is positive: 2πi * (k * n) / N
    exponent_cols = 2j * np.pi * k_indices * n_indices / width
    col_transformi_matrix = np.exp(exponent_cols)
    
    # --- Step 2: Create the Column-wise Inverse Matrix ---
    m_indices = np.arange(height)
    l_indices = m_indices.reshape((height, 1))
    
    # Exponent is positive: 2πi * (l * m) / M
    exponent_rows = 2j * np.pi * l_indices * m_indices / height
    idft_matrix_rows = np.exp(exponent_rows)
    
    # --- Step 3: Matrix Multiplication and Normalization ---
    # Apply transformation: U_inv @ F @ V_inv
    reconstructed_complex = idft_matrix_rows @ frequency_domain @ col_transformi_matrix
    
    # For IDFT, we must normalize by the total number of pixels (M * N)
    normalization_factor = height * width
    spatial_image = reconstructed_complex / normalization_factor
    
    # Return only the real component (imaginary part should be negligible/zero due to float precision)
    return spatial_image.real

# def main():
#     try:
#         # Load the image
#         img = Image.open('mona_lisa.jpg')
        
#         # Resize image for demonstration if it's too large
#         # (Even with optimization, O(N^3) is heavy for full HD images in raw Python)
#         img = img.resize((128, 128)) 
        
#         gray_scale = img.convert('L')
#         intensity_array_2d = np.array(gray_scale)

#         print(f"Processing Image of shape: {intensity_array_2d.shape}")

#         # 1. Forward Transform
#         frequency_domain_array = discrete_fourier_transform(intensity_array_2d)
#         print("\nFrequency Domain (First 5x5 corner):")
#         print(np.round(frequency_domain_array[:5, :5], 2))

#         # 2. Inverse Transform (Verification)
#         reconstructed_image = inverse_discrete_fourier_transform(frequency_domain_array)
        
#         # Verify correctness
#         is_close = np.allclose(intensity_array_2d, reconstructed_image, atol=1e-5)
#         print(f"\nReconstruction successful? {is_close}")
#         print("_______________________________________________________")

#         print(f"Low pass filter applied: {apply_low_pass_filter(frequency_domain_array, 10.0)}")
#         print("_______________________________________________________")
#         plot_magnitude_spectrum(frequency_domain_array)
#     except FileNotFoundError:
#         print("Error: 'mona_lisa.jpg' not found. Please ensure the image exists.")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")

# if __name__ == "__main__":
#     main()