import numpy as np
from scipy import signal

import correlate


import time

arr1 = np.random.randn(100000)
arr2 = np.random.randn(5)

scipytime = time.time()
out1 = signal.correlate(arr1, arr2, mode="valid", method="direct")
print(f"SCIPY TOOK {time.time() - scipytime}")

corrtime = time.time()
out2 = correlate.correlate1d(arr1, arr2)
print(f"CORR TOOK {time.time() - corrtime}")


print(out1[:10])
print(out2[:10])
