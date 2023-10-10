import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#should i filter anomalies? 
df = pd.read_csv('exp_ds.csv')
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df = df.sort_values('date')
df = df.set_index('date')
df = df.dropna()


# Apply FFT
n = len(df)
freq = np.fft.fftfreq(n) # generates arrow of 
fft_vals = np.fft.fft(df['qty']) # compute fft

amplitude = np.abs(fft_vals)
dominant_frequency = np.argmax(amplitude)
# Get absolute value to determine magnitude and normalize by the number of data points


# Identify major frequencies
threshold = np.max(fft_vals) * 0.20  # Adjust the threshold to catch major peaks
important_freqs = freq[np.where(fft_vals > threshold)]
print("Important frequencies: ", important_freqs)

# intresting frequency is 0.00459242 , T = 217 days, 

# plot initial data with sinus exp(-2pi * 0.00459242 * t) on top 
plt.figure(figsize=(12, 6))
plt.title("Sales Data")
plt.xlabel("Date")
plt.ylabel("Quantity")
plt.plot(freq, fft_vals, label="FFT")

plt.show()

time = np.arange(len(df))  
trig_function = np.cos(2 * np.pi * dominant_frequency * time / len(df))

plt.figure(figsize=(12, 6))
plt.title("Trigonometric Function with Dominant Frequency")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.plot(df.index, trig_function)
plt.show()





