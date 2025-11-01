import numpy as np

# Sampling parameters
fs = 1000          # Sampling frequency (Hz)
t_end = 1.0        # Signal duration (seconds)
t = np.arange(0, t_end, 1/fs)

# Create a clean 5 Hz sine wave
f_signal = 5
clean_signal = np.sin(2 * np.pi * f_signal * t)

# Add high-frequency noise
noise = 0.5 * np.random.randn(len(t))          # white noise
noise += 0.3 * np.sin(2 * np.pi * 200 * t)     # 200 Hz sine
noise += 0.2 * np.sin(2 * np.pi * 350 * t)     # 350 Hz sine

# Combine
noisy_signal = clean_signal + noise

# Save to text file (one value per line)
np.savetxt("noisy_signal.txt", noisy_signal, fmt="%.6f")

print(f"Saved {len(noisy_signal)} samples to noisy_signal.txt")
