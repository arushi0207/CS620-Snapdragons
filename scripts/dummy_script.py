import numpy as np
from featurehub.native.fastcv_bridge import dummy_add_one

x = np.array([1, 2, 3], dtype=np.uint8)
print("before:", x)
y = dummy_add_one(x)
print("after:", y)
