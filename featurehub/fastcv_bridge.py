from __future__ import annotations

from ctypes import (
    CDLL,
    Structure,
    POINTER,
    c_uint8,
    c_int,
    c_float,
)
from pathlib import Path
from typing import List, Tuple

import sys
import numpy as np

# --------------------------------------
# Load the native fastcv_bridge library
# --------------------------------------

if sys.platform == "win32":
    _LIB_NAME = "fastcv_bridge.dll"
elif sys.platform == "linux":
    _LIB_NAME = "libfastcv_bridge.so"
elif sys.platform == "darwin":
    _LIB_NAME = "libfastcv_bridge.dylib"
else:
    raise RuntimeError(f"Unsupported platform: {sys.platform}")

_here = Path(__file__).resolve().parent
_lib_path = _here / _LIB_NAME
if not _lib_path.exists():
    raise FileNotFoundError(f"FastCV bridge library not found at {_lib_path}")

_lib = CDLL(str(_lib_path))


# --------------------------------------
# Optional: existing dummy_add_one
# --------------------------------------

# If you still have dummy_add_one in C, keep this:
try:
    _lib.dummy_add_one.argtypes = [POINTER(c_uint8), c_int]
    _lib.dummy_add_one.restype = None

    def dummy_add_one(buf: np.ndarray) -> None:
        if buf.dtype != np.uint8:
            raise ValueError("dummy_add_one expects uint8 array")
        ptr = buf.ctypes.data_as(POINTER(c_uint8))
        _lib.dummy_add_one(ptr, buf.size)

except AttributeError:
    # Function not in DLL, ignore
    pass


# --------------------------------------
# FastCV face detection bindings
# --------------------------------------

class FcvRect(Structure):
    _fields_ = [
        ("x", c_float),
        ("y", c_float),
        ("width", c_float),
        ("height", c_float),
    ]


# Make sure the symbol exists in the DLL before touching argtypes
try:
    _fastcv_detect_faces = _lib.fastcv_detect_faces
except AttributeError as e:
    raise AttributeError(
        "fastcv_detect_faces not found in fastcv_bridge library. "
        "Did you rebuild fastcv_bridge.c after adding the function?"
    ) from e

_fastcv_detect_faces.argtypes = [
    POINTER(c_uint8),   # gray image data
    c_int,              # width
    c_int,              # height
    POINTER(FcvRect),   # output rectangles
    c_int,              # max_faces
]
_fastcv_detect_faces.restype = c_int


def fastcv_detect_faces(
    gray: np.ndarray, max_faces: int = 8
) -> List[Tuple[float, float, float, float]]:
    """
    Run FastCV face detection on a grayscale uint8 image.
    Returns a list of (x, y, w, h) in pixel coordinates.
    """
    if gray.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {gray.shape}")
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8, copy=False)

    h, w = gray.shape
    rects = (FcvRect * max_faces)()

    count = _fastcv_detect_faces(
        gray.ctypes.data_as(POINTER(c_uint8)),
        w,
        h,
        rects,
        max_faces,
    )

    if count <= 0:
        return []

    count = min(count, max_faces)
    return [
        (float(r.x), float(r.y), float(r.width), float(r.height))
        for r in rects[:count]
    ]
