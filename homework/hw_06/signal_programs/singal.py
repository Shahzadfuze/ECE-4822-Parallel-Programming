import numpy as np

# Parameters
frequency = 5        # Hz
amplitude = 1
sampling_rate = 1000 # samples per second
duration = 1         # seconds

# Generate time vector
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Generate sine wave
y = amplitude * np.sin(2 * np.pi * frequency * t)

# Print header
print("Amplitude")

# Print first 50 samples (to avoid flooding the console)
for i in range(50):
#        print(f"{y[i]:.4f}")

        #Optional: print all samples (uncomment below if you want everything)
        for ti, yi in zip(t, y):
                print(f"{yi:.4f}")
    
