import numpy as np

def generate_giant_signal(filename='giant_signal.txt',
                          fs=1000,        # Sampling frequency in Hz
                          duration=120,    # Duration in seconds
                          freqs=[5, 50, 120],  # Sine wave frequencies in Hz
                          amplitudes=[0.5, 0.25, 0.1],  # Corresponding amplitudes
                          noise_std=0.05  # Standard deviation of Gaussian noise
):
    """
    Generates a giant signal composed of multiple sine waves and Gaussian noise
    and saves it to a text file.
    """
    num_samples = fs * duration
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # Initialize signal
    signal = np.zeros_like(t)

    # Add sine waves
    for amp, f in zip(amplitudes, freqs):
        signal += amp * np.sin(2 * np.pi * f * t)
    
        # Add Gaussian noise
        signal += np.random.normal(0, noise_std, size=num_samples)

        # Save to file (one sample per line)
        np.savetxt(filename, signal, fmt='%.6f')
    print(f"Giant signal saved to '{filename}' with {num_samples} samples.")
    
    # Example usage
if __name__ == "__main__":
    generate_giant_signal(filename='medium_signal.txt',
                          fs=10000,
                          duration=1,  # 5 minutes → 300,000 samples
                          freqs=[5, 50, 120],
                          amplitudes=[0.5, 0.25, 0.1],
                          noise_std=0.05)
                                                                        
