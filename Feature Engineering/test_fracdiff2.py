import pandas as pd
import numpy as np

def getWeights(d, size):
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    return np.array(w)

def fast_fracdiff(series, d, window=100):
    weights = getWeights(d, window)
    res = np.convolve(series.to_numpy(), weights, mode='full')[:len(series)]
    res[:window-1] = np.nan
    return pd.Series(res, index=series.index)

np.random.seed(42)
df = pd.DataFrame({'close': np.cumsum(np.random.randn(1000000))})
import time
t0 = time.time()
df['close_frac'] = fast_fracdiff(df['close'], 0.45, 100)
print(f"Time taken: {time.time() - t0:.2f}s")
print(df.head(105).tail())
