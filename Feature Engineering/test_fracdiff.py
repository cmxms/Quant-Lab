import pandas as pd
import numpy as np
from tsfracdiff import FractionalDifferentiator

np.random.seed(42)
df = pd.DataFrame({'close': np.cumsum(np.random.randn(1000))})
fd = FractionalDifferentiator() # Wait, how to set d and window?
print(dir(fd))
