import numpy as np

def reconstruct_with_random_phase(frequency_domain: np.array) -> np.array:
    """
    Reconstructs image using the real magnitude but completely random phase.
    This shows what happens when you keep the 'colors/textures' but lose the 'structure'.
    """
    # 1. Extract the original magnitude
    magnitude = np.abs(frequency_domain)
    
    # 2. Create a random phase array between -pi and pi
    # Same shape as the original image
    random_phase = np.random.uniform(-np.pi, np.pi, frequency_domain.shape)
    
    # 3. Combine: Magnitude * e^(i * phase)
    # This uses Euler's formula to build the new complex numbers
    synthetic_frequency = magnitude * np.exp(1j * random_phase)
    
    return synthetic_frequency

def reconstruct_with_flat_magnitude(frequency_domain: np.array) -> np.array:
    """
    Reconstructs image using the original phase but sets all magnitudes to 1.
    This shows what happens when you throw away the 'energy' but keep the 'structure'.
    """
    # 1. Extract the original phase
    # np.angle returns the phase of complex numbers in radians
    phase = np.angle(frequency_domain)
    
    # 2. Set magnitude to 1 for every single frequency
    flat_magnitude = np.ones(frequency_domain.shape)
    
    # 3. Combine: 1 * e^(i * phase)
    synthetic_frequency = flat_magnitude * np.exp(1j * phase)
    
    return synthetic_frequency