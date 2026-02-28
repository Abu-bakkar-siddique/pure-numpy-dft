import numpy as np
from PIL import Image
from transform import discrete_fourier_transform, inverse_discrete_fourier_transform
from utils import save_array_as_grayscale_jpeg
from filters import apply_low_pass_filter, apply_high_pass_filter
from plots import plot_magnitude_spectrum
from phase import reconstruct_with_random_phase, reconstruct_with_flat_magnitude
from raw_fourier import fourier_1D

def main():
    try:
        # Load the image
        img = Image.open('mona_lisa.jpg')
        
        # Resize image for demonstration if it's too large
        # (Even with optimization, O(N^3) is heavy for full HD images in raw Python)
        img = img.resize((128, 128))
        
        gray_scale = img.convert('L')
        intensity_array_2d = np.array(gray_scale)

        print(f"{'='*60}")
        print(f"DIGITAL IMAGE PROCESSING PROJECT  DEMO")
        print(f"{'='*60}")
        print(f"Processing Image of shape: {intensity_array_2d.shape}\n")

        # ==============================================================
        # 1. BASIC FOURIER TRANSFORM OPERATIONS
        # ==============================================================
        print(f"{'─'*60}")
        print("1. FOURIER TRANSFORM: Spatial → Frequency Domain")
        print(f"{'─'*60}")

        # 1A. Forward Transform
        frequency_domain_array = discrete_fourier_transform(intensity_array_2d)
        print("✓ discrete_fourier_transform() - Converted image to frequency domain")
        print(f"  Frequency Domain (First 5x5 corner):")
        print(f"  {np.round(frequency_domain_array[:5, :5], 2)}\n")

        # 1B. Inverse Transform (Verification)
        print("Testing Inverse Transform...")
        reconstructed_image = inverse_discrete_fourier_transform(frequency_domain_array)
        is_close = np.allclose(intensity_array_2d, reconstructed_image, atol=1e-5)
        print(f"✓ inverse_discrete_fourier_transform() - Frequency → Spatial Domain")
        print(f"  Reconstruction successful? {is_close}")
        save_array_as_grayscale_jpeg(np.abs(reconstructed_image), 'reconstructed_original.jpg')
        print(f"  Saved: reconstructed_original.jpg\n")

        # ==============================================================
        # 2. MAGNITUDE SPECTRUM VISUALIZATION
        # ==============================================================
        print(f"{'─'*60}")
        print("2. VISUALIZATION: Magnitude Spectrum Analysis")
        print(f"{'─'*60}")
        print("✓ plot_magnitude_spectrum() - Displaying frequency distribution...")
        print("  (2D Magnitude Spectrum + Radial Energy Profile)")
        plot_magnitude_spectrum(frequency_domain_array, plot_radial=True)
        print()

        # ==============================================================
        # 3. LOW PASS FILTERING
        # ==============================================================
        print(f"{'─'*60}")
        print("3. LOW PASS FILTER: Removing High Frequencies (Blur Effect)")
        print(f"{'─'*60}")
        print("✓ apply_low_pass_filter() with radius=20")
        low_pass_filtered = apply_low_pass_filter(frequency_domain_array, 20)
        low_pass_image = inverse_discrete_fourier_transform(low_pass_filtered)
        save_array_as_grayscale_jpeg(np.abs(low_pass_image), 'low_pass_filtered.jpg')
        print("  Saved: low_pass_filtered.jpg")
        
        print("✓ apply_low_pass_filter() with radius=10 (more blur)")
        low_pass_filtered_aggressive = apply_low_pass_filter(frequency_domain_array, 10)
        low_pass_image_aggressive = inverse_discrete_fourier_transform(low_pass_filtered_aggressive)
        save_array_as_grayscale_jpeg(np.abs(low_pass_image_aggressive), 'low_pass_aggressive.jpg')
        print("  Saved: low_pass_aggressive.jpg\n")

        # ==============================================================
        # 4. HIGH PASS FILTERING
        # ==============================================================
        print(f"{'─'*60}")
        print("4. HIGH PASS FILTER: Removing Low Frequencies (Edge Detection)")
        print(f"{'─'*60}")
        print("✓ apply_high_pass_filter() with radius=30")
        high_pass_filtered = apply_high_pass_filter(frequency_domain_array, 60)
        high_pass_image = inverse_discrete_fourier_transform(high_pass_filtered)
        save_array_as_grayscale_jpeg(np.abs(high_pass_image), 'high_pass_filtered.jpg')
        print("  Saved: high_pass_filtered.jpg")
        
        print("✓ apply_high_pass_filter() with radius=50 (finer edges)")
        high_pass_filtered_fine = apply_high_pass_filter(frequency_domain_array, 50)
        high_pass_image_fine = inverse_discrete_fourier_transform(high_pass_filtered_fine)
        save_array_as_grayscale_jpeg(np.abs(high_pass_image_fine), 'high_pass_fine.jpg')
        print("  Saved: high_pass_fine.jpg\n")

        # ==============================================================
        # 5. PHASE EXPERIMENTS
        # ==============================================================
        print(f"{'─'*60}")
        print("5. PHASE ANALYSIS: Demonstrating Phase vs Magnitude")
        print(f"{'─'*60}")
        
        # 5A. Random Phase (Keep Magnitude, Random Phase)
        print("✓ reconstruct_with_random_phase() - Keeping magnitude, random phase")
        random_phase_freq = reconstruct_with_random_phase(frequency_domain_array)
        random_phase_image = inverse_discrete_fourier_transform(random_phase_freq)
        save_array_as_grayscale_jpeg(np.abs(random_phase_image), 'random_phase.jpg')
        print("  Result: Noise image (structure lost!)")
        print("  Saved: random_phase.jpg")
        
        # 5B. Flat Magnitude (Original Phase, Magnitude = 1)
        print("✓ reconstruct_with_flat_magnitude() - Keeping phase, flat magnitude")
        flat_magnitude_freq = reconstruct_with_flat_magnitude(frequency_domain_array)
        flat_magnitude_image = inverse_discrete_fourier_transform(flat_magnitude_freq)
        save_array_as_grayscale_jpeg(np.abs(flat_magnitude_image), 'flat_magnitude.jpg')
        print("  Result: Structure preserved (magnitude loss minimal!)")
        print("  Saved: flat_magnitude.jpg\n")

        # ==============================================================
        # 6. 1D FOURIER TRANSFORM (Educational)
        # ==============================================================
        print(f"{'─'*60}")
        print("6. 1D FOURIER TRANSFORM: Basic Implementation")
        print(f"{'─'*60}")
        print("✓ fourier_1D() - Computing 1D DFT on sample vector")
        sample_vector = [1, 7, 3, 5]
        dft_1d_result = fourier_1D(sample_vector)
        print(f"  Input vector: {sample_vector}")
        print(f"  DFT output (first 2 components):")
        print(f"  X(0) = {dft_1d_result[0]:.2f}")
        print(f"  X(1) = {dft_1d_result[1]:.2f}\n")

        # ==============================================================
        # 7. UTILITY FUNCTION DEMONSTRATION
        # ==============================================================
        print(f"{'─'*60}")
        print("7. UTILITY FUNCTIONS: File I/O Operations")
        print(f"{'─'*60}")
        print("✓ save_array_as_grayscale_jpeg()")
        print("  All output images have been saved using this function:")
        print("  - reconstructed_original.jpg")
        print("  - low_pass_filtered.jpg")
        print("  - low_pass_aggressive.jpg")
        print("  - high_pass_filtered.jpg")
        print("  - high_pass_fine.jpg")
        print("  - random_phase.jpg")
        print("  - flat_magnitude.jpg\n")

        # ==============================================================
        # SUMMARY
        # ==============================================================
        print(f"{'='*60}")
        print("SUMMARY: All Functions Successfully Executed")
        print(f"{'='*60}")
        print("\nFunctions Called:")
        print("  ✓ discrete_fourier_transform()")
        print("  ✓ inverse_discrete_fourier_transform()")
        print("  ✓ plot_magnitude_spectrum()")
        print("  ✓ apply_low_pass_filter()")
        print("  ✓ apply_high_pass_filter()")
        print("  ✓ reconstruct_with_random_phase()")
        print("  ✓ reconstruct_with_flat_magnitude()")
        print("  ✓ fourier_1D()")
        print("  ✓ save_array_as_grayscale_jpeg()")
        print(f"\nOutput files saved in current directory")
        print(f"For detailed documentation, see: PROJECT_DOCUMENTATION.md")
        print(f"{'='*60}\n")

    except FileNotFoundError:
        print("Error: 'mona_lisa.jpg' not found. Please ensure the image exists.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()