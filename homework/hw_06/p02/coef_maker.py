from scipy import signal

fs = 1000  # sample rate
fc = 50    # cutoff frequency
b, a = signal.butter(2, fc/(fs/2))  # 2nd-order lowpass Butterworth
print(b)
print(a)
