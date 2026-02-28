# FourierVision: Phase Analysis, Magnitude Spectrum & Raw Image Filters - Pure Numpy DFT Implementation (No OpenCV)

**Project Name:** FourierVision  
**Subtitle:** Raw Fourier Transform Analysis Using Pure NumPy Computations

## Overview
This project implements **pure NumPy-based** Digital Image Processing techniques **without any OpenCV dependency**, focusing on **Fourier Transform analysis**, **phase-magnitude decomposition**, and **raw frequency domain filtering**. The codebase provides foundational tools for transforming images between spatial and frequency domains using direct matrix multiplication, applying custom filters, and visualizing frequency components through magnitude spectrum analysis.

---

## 🔬 Project Philosophy

> **This project demonstrates pure mathematical implementations WITHOUT relying on black-box libraries like OpenCV.**
> 
> Every Fourier Transform is computed through **raw matrix multiplication**, not FFT algorithms.  
> Every filter is built from **first-principles mathematics**, not library presets.  
> Every visualization exposes the **raw magnitude spectrum**, revealing exactly what each frequency component contributes.

### Key Differentiators:
- ✅ **Pure NumPy**: No OpenCV, scikit-image, or other image processing libraries
- ✅ **Educational**: Every computation shows the mathematical formula explicitly
- ✅ **Phase-Magnitude Study**: Demonstrates the critical role of phase information
- ✅ **Raw Filters**: Implements ideal low-pass and high-pass filters from scratch
- ✅ **Magnitude Spectrum Analysis**: Visualizes frequency distribution with radial energy profiles

---

## Module: `transform.py`
**Purpose:** Contains core DFT and IDFT implementations for image processing.

### Function: `discrete_fourier_transform(grayscale_image: np.array) -> np.array`

**What it does:**
Converts a grayscale image from the spatial domain (pixel values) to the frequency domain (complex Fourier coefficients). This transformation reveals the underlying frequency components that make up the image.

**How it works:**
1. Takes a 2D grayscale image (height × width matrix of pixel intensities)
2. Creates two transform matrices:
   - **Row Transform Matrix (Width × Width):** Uses the DFT formula with exponents `-2πi(k*n)/width`
   - **Column Transform Matrix (Height × Height):** Uses the DFT formula with exponents `-2πi(l*m)/height`
3. Applies separable 2D DFT using matrix multiplication: `U @ I @ V`
4. Returns a 2D array of complex numbers where:
   - Magnitude represents the "strength" of each frequency component
   - Phase represents the "location" of features at that frequency

**Mathematical Formula:**
$$F(k, l) = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} f(m,n) \cdot e^{-2\pi i(km/M + ln/N)}$$

**Parameters:**
- `grayscale_image`: 2D numpy array of pixel intensities (0-255)

**Returns:**
- 2D numpy array of complex numbers

---

### Function: `inverse_discrete_fourier_transform(frequency_domain: np.array) -> np.array`

**What it does:**
Converts the frequency domain representation back to the spatial domain, reconstructing the original (or filtered) image from its Fourier coefficients.

**How it works:**
1. Creates inverse transform matrices with positive exponents (`+2πi` instead of `-2πi`)
2. Applies the inverse transformation: `U_inv @ F @ V_inv`
3. Normalizes the result by dividing by the total number of pixels (M × N)
4. Extracts the real component (imaginary part should be negligible due to float precision)
5. Returns the reconstructed spatial domain image

**Mathematical Formula:**
$$f(m, n) = \frac{1}{M \cdot N} \sum_{k=0}^{M-1} \sum_{l=0}^{N-1} F(k,l) \cdot e^{2\pi i(km/M + ln/N)}$$

**Parameters:**
- `frequency_domain`: 2D numpy array of complex numbers (DFT output)

**Returns:**
- 2D numpy array of real numbers representing the reconstructed image

---

## Module: `filters.py`
**Purpose:** Provides frequency domain filtering operations.

### Function: `apply_low_pass_filter(frequency_domain: np.array, radius: float) -> np.array`

**What it does:**
Applies an Ideal Low Pass Filter (ILPF) that preserves low-frequency components while removing high-frequency components, resulting in a blurred/smoothed image.

**How it works:**
1. Shifts the zero-frequency component to the center using `np.fft.fftshift()`
2. Creates a 2D coordinate grid to calculate distances from the center frequency
3. Generates a circular mask:
   - Value 1 for all pixels within the radius (keep these frequencies)
   - Value 0 for all pixels outside the radius (remove these frequencies)
4. Multiplies the frequency domain data by the mask (element-wise)
5. Shifts back to original positions using `np.fft.ifftshift()`

**Visual Effect:**
- Reduces high-frequency noise and fine details
- Creates a blurred effect
- Smaller radius = more blur

**Parameters:**
- `frequency_domain`: 2D numpy array of complex numbers (DFT output)
- `radius`: Float value representing the cutoff distance in frequency space

**Returns:**
- Filtered 2D numpy array of complex numbers

---

### Function: `apply_high_pass_filter(frequency_domain: np.array, radius: float) -> np.array`

**What it does:**
Applies an Ideal High Pass Filter (IHPF) that removes low-frequency components while preserving high-frequency components, highlighting edges and fine details.

**How it works:**
1. Shifts the zero-frequency component to the center
2. Creates a 2D coordinate grid for distance calculations
3. Generates a "donut" mask:
   - Value 0 for all pixels within the radius (remove low frequencies)
   - Value 1 for all pixels outside the radius (keep high frequencies)
4. Applies the mask via element-wise multiplication
5. Shifts back to original positions

**Visual Effect:**
- Highlights edges and boundaries
- Removes smooth areas and low-frequency noise
- Larger radius = only finest edges remain

**Parameters:**
- `frequency_domain`: 2D numpy array of complex numbers (DFT output)
- `radius`: Float value representing the cutoff distance in frequency space

**Returns:**
- Filtered 2D numpy array of complex numbers

---

## Module: `phase.py`
**Purpose:** Experimental phase-magnitude decomposition studies to demonstrate the critical information carried by each component.

### Conceptual Introduction

This module explores one of the most profound discoveries in image processing:

> **Phase information encodes spatial structure; Magnitude information encodes energy/intensity.**

Without this module's experiments, this would be just theory. Here, we **prove it empirically** by manipulating the phase and magnitude independently and observing the consequences.

### Function: `reconstruct_with_random_phase(frequency_domain: np.array) -> np.array`

**What it does:**
Reconstructs an image using the original magnitude spectrum but replaces the phase information with random values. This demonstrates that while magnitude carries "energy/color," phase carries the "structure/spatial information."

**How it works:**
1. Extracts the magnitude (absolute value) from each frequency component
2. Generates random phase angles uniformly distributed between -π and π
3. Reconstructs complex numbers using Euler's formula: `magnitude * e^(i*phase)`
4. Returns the modified frequency domain array

**Mathematical Formula:**
$$F_{synthetic}(k,l) = |F(k,l)| \cdot e^{i\theta_{random}}$$

where $\theta_{random}$ is uniformly distributed in $[-\pi, \pi]$

**Observation:**
The resulting image appears as noise despite containing the same "energy" as the original, proving that phase information encodes spatial structure.

**Parameters:**
- `frequency_domain`: 2D numpy array of complex numbers (DFT output)

**Returns:**
- 2D numpy array of complex numbers with preserved magnitude but random phase

---

### Function: `reconstruct_with_flat_magnitude(frequency_domain: np.array) -> np.array`

**What it does:**
Reconstructs an image by preserving the original phase but setting all magnitudes to 1. This demonstrates that phase alone is sufficient to preserve image structure.

**How it works:**
1. Extracts the phase angle from each frequency component using `np.angle()`
2. Creates a uniform magnitude array with all values set to 1
3. Reconstructs complex numbers using: `1 * e^(i*phase)`
4. Returns the modified frequency domain array

**Mathematical Formula:**
$$F_{synthetic}(k,l) = 1 \cdot e^{i\angle F(k,l)} = e^{i\angle F(k,l)}$$

**Observation:**
The resulting image retains all structural details despite losing magnitude information, proving that phase carries complete positional information.

**Parameters:**
- `frequency_domain`: 2D numpy array of complex numbers (DFT output)

**Returns:**
- 2D numpy array of complex numbers with magnitude 1 and original phase

---

## Module: `plots.py`
**Purpose:** Raw magnitude spectrum visualization - exposing the frequency domain without abstraction.

### Why Magnitude Spectrum Matters

The magnitude spectrum is the **visual manifestation of the Fourier Transform**. It shows:
- Where in the frequency space the image's "energy" is concentrated
- Which frequencies dominate the image structure
- How filters will affect different regions of the spectrum

This is fundamental knowledge that libraries like OpenCV hide behind convenience functions. Here, we compute and visualize it raw.

### Function: `plot_magnitude_spectrum(frequency_domain: np.array, plot_radial: bool = True)`

**What it does:**
Visualizes the Magnitude Spectrum of a frequency domain image, showing the distribution of frequency components. Optionally includes a radial energy profile showing how information is distributed across frequency distances.

**How it works:**

**Main Plot (2D Magnitude Spectrum):**
1. Shifts zero-frequency component to center using `np.fft.fftshift()`
2. Computes magnitude (absolute value) of each complex frequency component
3. Applies log scaling: `log(1 + magnitude)` to compress the dynamic range (DC component is often 1,000,000× larger than high frequencies)
4. Displays as grayscale image with colorbar

**Optional Radial Energy Plot (1D Energy Profile):**
1. Calculates distance of every pixel from the center frequency
2. Bins pixels by their distance (converts to integer radii)
3. Sums magnitude energy within each radius bin
4. Plots as 1D graph showing how energy decreases with frequency

**Visual Interpretation:**
- Bright center: Low frequencies (average brightness, smooth areas)
- Bright edges: High frequencies (edges, fine details)
- Log scale reveals details that would otherwise be invisible due to dynamic range

**Parameters:**
- `frequency_domain`: 2D numpy array of complex numbers (DFT output)
- `plot_radial`: Boolean flag to enable/disable radial energy profile (default: True)

**Returns:**
- None (displays matplotlib plots)

---

## Module: `utils.py`
**Purpose:** Utility functions for image file I/O.

### Function: `save_array_as_grayscale_jpeg(intensity_array: np.array, filename: str = "output.jpg")`

**What it does:**
Converts a 2D numpy array of pixel intensities into a grayscale JPEG image file and saves it to disk.

**How it works:**
1. **Clipping:** Uses `np.clip()` to constrain all values to [0, 255]
   - Values < 0 become 0
   - Values > 255 become 255
   - This is necessary because JPEG requires valid pixel intensities
2. **Type Conversion:** Converts to `uint8` (unsigned 8-bit integer) format required by PIL
3. **Image Creation:** Creates a PIL Image object in 'L' mode (Luminance/grayscale)
4. **File Saving:** Saves the image as JPEG format to the specified filename

**Parameters:**
- `intensity_array`: 2D numpy array of pixel values (float or int)
- `filename`: String path/filename for output JPEG (default: "output.jpg")

**Returns:**
- None (prints confirmation message)

---

## Module: `raw_fourier.py`
**Purpose:** Basic 1D Fourier Transform implementation (educational).

### Function: `fourier_1D(input_vector: list) -> list`

**What it does:**
Computes the 1D Discrete Fourier Transform of an input vector using nested loops. This is a direct, unoptimized implementation that shows the mathematical formula explicitly.

**How it works:**
1. Iterates through each output frequency index `k` from 0 to N-1
2. For each `k`, iterates through each input sample index `n` from 0 to N-1
3. Computes the phase angle: `-2π*k*n/N`
4. Multiplies each input sample by `e^(i*angle)` and accumulates the sum
5. Stores the complex result in the output vector

**Mathematical Formula:**
$$X(k) = \sum_{n=0}^{N-1} x(n) \cdot e^{-2\pi i \cdot k \cdot n / N}$$

**Time Complexity:** O(N²) - very slow for large inputs. NumPy's FFT uses FFT algorithm with O(N log N) complexity.

**Parameters:**
- `input_vector`: Python list of numeric values

**Returns:**
- Python list of complex numbers (Fourier coefficients)

---

## Module: `img_processing.py`
**Purpose:** Simple image loading demonstration.

### Function: `main()`

**What it does:**
Loads a JPEG image, converts it to grayscale, and prints the resulting pixel intensity array.

**How it works:**
1. Opens 'mona_lisa.jpg' using PIL
2. Converts to grayscale ('L' mode) 
3. Converts to numpy array
4. Prints the array to console

---

## Data Flow in the Project

```
Input Image (JPEG)
       ↓
load image & convert to grayscale
       ↓
discrete_fourier_transform()
       ↓
Frequency Domain (Complex Numbers)
       ├→ plot_magnitude_spectrum() [Visualization]
       ├→ apply_low_pass_filter() → inverse_discrete_fourier_transform() [Blurred Image]
       ├→ apply_high_pass_filter() → inverse_discrete_fourier_transform() [Edge Detection]
       ├→ reconstruct_with_random_phase() → inverse_discrete_fourier_transform() [Noise]
       └→ reconstruct_with_flat_magnitude() → inverse_discrete_fourier_transform() [Structure only]
       ↓
save_array_as_grayscale_jpeg() [Output Files]
```

---

## Key Concepts

### Fourier Transform
Decomposes an image into its frequency components. Low frequencies = smooth areas. High frequencies = edges/details.

### Phase vs Magnitude
- **Magnitude:** Energy/brightness of each frequency component
- **Phase:** Location/position of features in the image
- Both are essential for perfect reconstruction

### Filtering in Frequency Domain
Much simpler than spatial domain convolution - just multiply by a mask!

### Why Log Scale?
The DC component (average brightness) can be 1,000,000× larger than fine details. Log scale makes fine details visible.

---

## Usage Example

```python
from transform import discrete_fourier_transform, inverse_discrete_fourier_transform
from filters import apply_low_pass_filter, apply_high_pass_filter
from plots import plot_magnitude_spectrum
from utils import save_array_as_grayscale_jpeg
from phase import reconstruct_with_random_phase, reconstruct_with_flat_magnitude
import numpy as np
from PIL import Image

# Load and prepare image
img = Image.open('mona_lisa.jpg')
img = img.resize((128, 128))
gray_scale = img.convert('L')
intensity_array = np.array(gray_scale)

# Transform to frequency domain
freq_domain = discrete_fourier_transform(intensity_array)

# Visualize
plot_magnitude_spectrum(freq_domain)

# Apply filters
low_pass = apply_low_pass_filter(freq_domain, 20)
high_pass = apply_high_pass_filter(freq_domain, 50)

# Reconstruct
low_pass_image = inverse_discrete_fourier_transform(low_pass)
high_pass_image = inverse_discrete_fourier_transform(high_pass)

# Phase experiments
random_phase = reconstruct_with_random_phase(freq_domain)
flat_magnitude = reconstruct_with_flat_magnitude(freq_domain)

# Save results
save_array_as_grayscale_jpeg(low_pass_image, 'low_pass.jpg')
save_array_as_grayscale_jpeg(high_pass_image, 'high_pass.jpg')
```

