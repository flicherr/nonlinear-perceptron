import numpy as np
import csv
import matplotlib.pyplot as plt

def generate_regression_data(
        N=200,
        x_min=-5.0,
        x_max=5.0,
        trend_slope=None,
        freq=None,
        noise_std=0.1,
        seed=None):
    """
    y = trend * x + sin(freq * x + phase) + noise
    y = 0.4x + sin(1.5x)
    """
    if seed is not None:
        np.random.seed(seed)

    x = np.linspace(x_min, x_max, N)

    if trend_slope is None:
        trend_slope = np.random.uniform(0.2, 0.6)
    if freq is None:
        freq = np.random.uniform(1.0, 2.0)
    phase = np.random.uniform(0, 2*np.pi)

    noise = np.random.normal(0, noise_std, N)

    y = trend_slope * x + np.sin(freq * x + phase) + noise

    return x, y, trend_slope, freq, phase


x, y, slope, freq, phase = generate_regression_data(N=200, noise_std=0.15)

with open("dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x", "y"])
    for xi, yi in zip(x, y):
        writer.writerow([xi, yi])

print(f"Generated dataset.csv with trend={slope:.3f}, freq={freq:.3f}, phase={phase:.3f}")

plt.figure(figsize=(8,4))
plt.scatter(x, y, color='blue', s=15, label='Samples')
plt.title('Generated Regression Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()