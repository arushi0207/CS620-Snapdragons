from ctypes import (
    CDLL,
    Structure,
    POINTER,
    c_uint8,
    c_int,
    c_int32,
    c_uint32,
)
import numpy as np
from pathlib import Path
import sys

# Load DLL from featurehub/native/
if sys.platform == "win32":
    _LIB_NAME = "fastcv_bridge.dll"
elif sys.platform == "linux":
    _LIB_NAME = "libfastcv_bridge.so"
elif sys.platform == "darwin":
    _LIB_NAME = "libfastcv_bridge.dylib"
else:
    raise RuntimeError(f"Unsupported platform: {sys.platform}")

_here = Path(__file__).resolve().parent
_native_dir = _here / "native"
_lib_path = _native_dir / _LIB_NAME

if not _lib_path.exists():
    raise FileNotFoundError(f"FastCV bridge library not found at: {_lib_path}")

_lib = CDLL(str(_lib_path))


# -----------------------------
# STRUCT (must match the C file)
# -----------------------------
class FcvRect(Structure):
    _fields_ = [
        ("x", c_int32),
        ("y", c_int32),
        ("width", c_uint32),
        ("height", c_uint32),
    ]


# -----------------------------
# BIND NATIVE FUNCTION
# -----------------------------
try:
    _lib.fastcv_detect_faces.argtypes = [
        POINTER(c_uint8),
        c_int,
        c_int,
        POINTER(FcvRect),
        c_int,
    ]
    _lib.fastcv_detect_faces.restype = c_int
except AttributeError:
    raise RuntimeError("fastcv_detect_faces not found in DLL.")


# -----------------------------
# PYTHON WRAPPER
# -----------------------------
def fastcv_detect_faces(gray: np.ndarray, max_faces: int = 8):
    """Call the FastCV C face-detection function."""
    if not isinstance(gray, np.ndarray):
        raise TypeError("Input must be a numpy array.")

    if gray.ndim != 2:
        raise ValueError("Expected a grayscale HxW image.")

    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)

    h, w = gray.shape
    rects = (FcvRect * max_faces)()

    count = _lib.fastcv_detect_faces(
        gray.ctypes.data_as(POINTER(c_uint8)),
        w,
        h,
        rects,
        max_faces,
    )

    return rects[:count]
