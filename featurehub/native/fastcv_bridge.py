# featurehub/native/fastcv_bridge.py
import os
import numpy as np
from ctypes import cdll, c_int, c_uint8, c_void_p

# Path to the shared library
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.path.join(_THIS_DIR, "fastcv_bridge.dll")  # <-- no extra "native"

if not os.path.exists(_LIB_PATH):
    raise FileNotFoundError(f"fastcv_bridge.dll not found at: {_LIB_PATH}")

_lib = cdll.LoadLibrary(_LIB_PATH)

# Configure the C function signature
_lib.dummy_add_one.argtypes = [c_void_p, c_int]
_lib.dummy_add_one.restype = None


def dummy_add_one(arr: np.ndarray) -> np.ndarray:
    """
    For testing: adds 1 to every byte of the array in-place via C.
    """
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    ptr = arr.ctypes.data_as(c_void_p)
    length = arr.size  # number of elements (bytes)

    _lib.dummy_add_one(ptr, c_int(length))
    return arr
