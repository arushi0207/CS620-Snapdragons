import os
import numpy as np
from ctypes import cdll, c_int, c_void_p

# Load the DLL
_LIB_PATH = os.path.join(os.path.dirname(__file__), "native", "fastcv_bridge.dll")
_lib = cdll.LoadLibrary(_LIB_PATH)

# --- dummy_add_one (still useful for testing) ---
_lib.dummy_add_one.argtypes = [c_void_p, c_int]
_lib.dummy_add_one.restype = None


def dummy_add_one(arr: np.ndarray) -> np.ndarray:
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    ptr = arr.ctypes.data_as(c_void_p)
    _lib.dummy_add_one(ptr, c_int(arr.size))
    return arr


# --- bgr_to_rgb_crop_resize_fastcv wrapper ---

_lib.bgr_to_rgb_crop_resize_fastcv.argtypes = [
    c_void_p,       # src pointer
    c_int, c_int,   # src_w, src_h
    c_int, c_int,   # crop_x, crop_y
    c_int, c_int,   # crop_w, crop_h
    c_void_p,       # dst pointer
    c_int, c_int,   # dst_w, dst_h
]
_lib.bgr_to_rgb_crop_resize_fastcv.restype = None


def fastcv_crop_resize_bgr_to_rgb(
    frame_bgr: np.ndarray,
    crop_x: int,
    crop_y: int,
    crop_w: int,
    crop_h: int,
    dst_w: int,
    dst_h: int,
) -> np.ndarray:
    """
    frame_bgr: H x W x 3, BGR, uint8 (from OpenCV)
    returns: dst_h x dst_w x 3, RGB, uint8
    """
    assert frame_bgr.dtype == np.uint8
    assert frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3

    if not frame_bgr.flags["C_CONTIGUOUS"]:
        frame_bgr = np.ascontiguousarray(frame_bgr)

    src_h, src_w, _ = frame_bgr.shape
    dst = np.empty((dst_h, dst_w, 3), dtype=np.uint8)

    _lib.bgr_to_rgb_crop_resize_fastcv(
        frame_bgr.ctypes.data_as(c_void_p),
        c_int(src_w), c_int(src_h),
        c_int(crop_x), c_int(copy_y := crop_y),  # Python 3.8 safe trick
        c_int(crop_w), c_int(crop_h),
        dst.ctypes.data_as(c_void_p),
        c_int(dst_w), c_int(dst_h),
    )

    return dst
