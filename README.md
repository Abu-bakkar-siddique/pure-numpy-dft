# Phase Analysis, Magnitude Spectrum & Raw Image Filters
**Pure NumPy DFT Implementation (No OpenCV)**

---

## Overview

This project implements pure NumPy-based Digital Image Processing techniques without any OpenCV dependency, focusing on Fourier Transform analysis, phase-magnitude decomposition, and raw frequency domain filtering. The codebase provides foundational tools for transforming images between spatial and frequency domains using direct matrix multiplication, applying custom filters, and visualizing frequency components through magnitude spectrum analysis.

---

## Project Philosophy

This project demonstrates pure mathematical implementations WITHOUT relying on black-box libraries like OpenCV.

Every Fourier Transform is computed through raw matrix multiplication, not FFT algorithms.  
Every filter is built from first-principles mathematics, not library presets.  
Every visualization exposes the raw magnitude spectrum, revealing exactly what each frequency component contributes.

**Key Differentiators:**
- Pure NumPy: No OpenCV, scikit-image, or other image processing libraries
- Educational: Every computation shows the mathematical formula explicitly
- Phase-Magnitude Study: Demonstrates the critical role of phase information
- Raw Filters: Implements ideal low-pass and high-pass filters from scratch
- Magnitude Spectrum Analysis: Visualizes frequency distribution with radial energy profiles

---

## Data Flow

```
Input Image (JPEG)
       |
load image & convert to grayscale
       |
discrete_fourier_transform()
       |
Frequency Domain (Complex Numbers)
       |-> plot_magnitude_spectrum()                                         [Visualization]
       |-> apply_low_pass_filter()  -> inverse_discrete_fourier_transform() [Blurred Image]
       |-> apply_high_pass_filter() -> inverse_discrete_fourier_transform() [Edge Detection]
       |-> reconstruct_with_random_phase() -> inverse_discrete_fourier_transform() [Noise]
       |-> reconstruct_with_flat_magnitude() -> inverse_discrete_fourier_transform() [Structure only]
       |
save_array_as_grayscale_jpeg()                                              [Output Files]
```

---

## Module: `transform.py`

Contains core DFT and IDFT implementations for image processing.

### `discrete_fourier_transform(grayscale_image: np.array) -> np.array`

Converts a grayscale image from the spatial domain to the frequency domain. Constructs two DFT basis matrices and applies separable 2D DFT via matrix multiplication: `U @ I @ V`.

- Row Transform Matrix (Width x Width): exponents `-2pi*i*(k*n)/width`
- Column Transform Matrix (Height x Height): exponents `-2pi*i*(l*m)/height`

**Formula:**

$$F(k, l) = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} f(m,n) \cdot e^{-2\pi i(km/M + ln/N)}$$

**Parameters:**
- `grayscale_image`: 2D numpy array of pixel intensities (0-255)

**Returns:** 2D numpy array of complex numbers

---

### `inverse_discrete_fourier_transform(frequency_domain: np.array) -> np.array`

Converts the frequency domain representation back to the spatial domain. Uses positive exponents (`+2pi*i`), applies inverse transformation `U_inv @ F @ V_inv`, normalizes by dividing by M x N, and returns the real component.

**Formula:**

$$f(m, n) = \frac{1}{M \cdot N} \sum_{k=0}^{M-1} \sum_{l=0}^{N-1} F(k,l) \cdot e^{2\pi i(km/M + ln/N)}$$

**Parameters:**
- `frequency_domain`: 2D numpy array of complex numbers

**Returns:** 2D numpy array of real numbers representing the reconstructed image

---

## Module: `filters.py`

Provides frequency domain filtering operations.

### `apply_low_pass_filter(frequency_domain: np.array, radius: float) -> np.array`

Applies an Ideal Low Pass Filter (ILPF). Shifts zero-frequency to center, builds a circular mask (1 inside radius, 0 outside), multiplies element-wise, then shifts back.

**Visual Effect:** Blurs the image. Smaller radius = more blur.

**Parameters:**
- `frequency_domain`: 2D numpy array of complex numbers
- `radius`: Cutoff distance in frequency space

**Returns:** Filtered 2D numpy array of complex numbers

---

### `apply_high_pass_filter(frequency_domain: np.array, radius: float) -> np.array`

Applies an Ideal High Pass Filter (IHPF). Same process as low-pass but with an inverted mask (0 inside radius, 1 outside).

**Visual Effect:** Highlights edges. Larger radius = only finest edges remain.

**Parameters:**
- `frequency_domain`: 2D numpy array of complex numbers
- `radius`: Cutoff distance in frequency space

**Returns:** Filtered 2D numpy array of complex numbers

---

## Module: `phase.py`

Experimental phase-magnitude decomposition studies.

### Conceptual Introduction

Phase information encodes spatial structure; magnitude information encodes energy/intensity. This module proves it empirically by manipulating each independently.

---

### `reconstruct_with_random_phase(frequency_domain: np.array) -> np.array`

Keeps original magnitude but replaces phase with random values uniform in [-pi, pi]. Reconstructs using Euler's formula: `magnitude * e^(i*random_phase)`.

**Formula:**

$$F_{synthetic}(k,l) = |F(k,l)| \cdot e^{i\theta_{random}}$$

**Observation:** Result appears as noise despite containing the same energy, proving phase encodes spatial structure.

**Parameters:**
- `frequency_domain`: 2D numpy array of complex numbers

**Returns:** 2D numpy array with preserved magnitude but random phase

---

### `reconstruct_with_flat_magnitude(frequency_domain: np.array) -> np.array`

Preserves original phase but sets all magnitudes to 1. Reconstructs using `e^(i*phase)`.

**Formula:**

$$F_{synthetic}(k,l) = e^{i\angle F(k,l)}$$

**Observation:** Structural details are retained despite losing all magnitude information, proving phase carries complete positional information.

**Parameters:**
- `frequency_domain`: 2D numpy array of complex numbers

**Returns:** 2D numpy array with magnitude 1 and original phase

---

## Module: `plots.py`

Raw magnitude spectrum visualization.

### `plot_magnitude_spectrum(frequency_domain: np.array, plot_radial: bool = True)`

Visualizes the magnitude spectrum of a frequency domain image.

**Main Plot:** Shifts zero-frequency to center, computes magnitude, applies log scaling `log(1 + magnitude)` to compress dynamic range, displays as grayscale with colorbar.

**Radial Energy Plot (optional):** Calculates distance of every pixel from center, bins by distance, sums energy per bin, plots as 1D profile.

**Parameters:**
- `frequency_domain`: 2D numpy array of complex numbers
- `plot_radial`: Enable/disable radial energy profile (default: True)

**Returns:** None (displays matplotlib plots)

---

## Module: `utils.py`

Utility functions for image file I/O.

### `save_array_as_grayscale_jpeg(intensity_array: np.array, filename: str = "output.jpg")`

Clips values to [0, 255], converts to uint8, creates a PIL Image in 'L' mode, and saves as JPEG.

**Parameters:**
- `intensity_array`: 2D numpy array of pixel values
- `filename`: Output path (default: "output.jpg")

**Returns:** None

---

## Module: `raw_fourier.py`

Basic 1D Fourier Transform implementation (educational).

### `fourier_1D(input_vector: list) -> list`

Computes the 1D DFT using nested loops. A direct, unoptimized implementation that maps exactly to the mathematical formula.

**Formula:**

$$X(k) = \sum_{n=0}^{N-1} x(n) \cdot e^{-2\pi i \cdot k \cdot n / N}$$

**Time Complexity:** O(N^2). NumPy's FFT runs at O(N log N).

**Parameters:**
- `input_vector`: Python list of numeric values

**Returns:** Python list of complex numbers (Fourier coefficients)

---

## Module: `img_processing.py`

### `main()`

Loads a grayscale image and prints the resulting pixel intensity array to console.

---

## Key Concepts

**Fourier Transform:** Decomposes an image into its frequency components. Low frequencies correspond to smooth areas; high frequencies correspond to edges and fine details.

**Phase vs Magnitude:** Magnitude carries the energy/brightness of each frequency component. Phase carries the location and structure of features. Both are required for perfect reconstruction, but phase is dominant for spatial coherence.

**Filtering in Frequency Domain:** Convolution in the spatial domain equals multiplication in the frequency domain (Convolution Theorem). This makes frequency domain filtering trivial: just multiply by a mask.

**Why Log Scale:** The DC component (average brightness) can be orders of magnitude larger than high frequency components. Log scaling compresses this range and makes fine details visible in the spectrum.

**Why This is Slow:** The matrix multiplication approach is O(N^4) for an N x N image. NumPy's `np.fft.fft2` uses the Fast Fourier Transform algorithm at O(N^2 log N). This project intentionally avoids FFT to keep the mathematics explicit.