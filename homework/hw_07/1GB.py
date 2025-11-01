import numpy as np

# Parameters
filename = "sine_signal.txt"
num_lines = 111_111_111  # ~1 GB
batch_size = 1_000_000   # Write in batches of 1 million lines
amplitude = 1.0
frequency = 1.0  # 1 full sine cycle across the whole file

# Generate indices in batches and write to file
with open(filename, "w") as f:
    for start in range(0, num_lines, batch_size):
        end = min(start + batch_size, num_lines)
        i = np.arange(start, end)
        values = amplitude * np.sin(2 * np.pi * frequency * i / num_lines)
        # Convert to string and add newline
        f.write("\n".join(f"{v:.6f}" for v in values) + "\n")
                                                    
