import numpy as np
from chaos.chaotic_features import compute_lyapunov_rosenstein

x = np.random.randn(1000)
print(compute_lyapunov_rosenstein(
    x, fs=5.0, mean_period=1.0,
    tau=5, m=4, slope_ros=[0.2, 4.0],
))
