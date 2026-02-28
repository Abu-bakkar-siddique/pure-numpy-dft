import numpy as np
from PIL import Image

def save_array_as_grayscale_jpeg(intensity_array: np.array, filename: str = "output.jpg"):
    """
    Converts a 2D numpy array of intensities into a grayscale JPEG image.
    
    Args:
        intensity_array: 2D numpy array (float or int)
        filename: The string path/name of the file to save
    """
    
    # 1. Handle Negative Values and Clipping
    # Intensities must be between 0 and 255. 
    # np.clip ensures any value < 0 becomes 0, and > 255 becomes 255.
    clipped_array = np.clip(intensity_array, 0, 255)
    
    # 2. Data Type Conversion
    # JPEG images require "unsigned 8-bit integers" (uint8).
    uint8_array = clipped_array.astype(np.uint8)
    
    # 3. Create PIL Image Object
    # 'L' mode stands for Luminance (standard 8-bit grayscale)
    img = Image.fromarray(uint8_array, mode='L')

    
    # 4. Save to disk
    img.save(filename, "JPEG")
    print(f"Image successfully saved as {filename}")

# --- Example Usage ---
# random_data = np.random.randint(0, 256, (256, 256))
# save_array_as_grayscale_jpeg(random_data, "test_image.jpg")