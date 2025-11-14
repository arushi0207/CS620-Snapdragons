#include <stdint.h>
#include "fastcv.h"   // from SDK/fastcv_sdk/inc

#ifdef __cplusplus
extern "C" {
#endif

// Keep this around if you still use it in tests
__declspec(dllexport)
void dummy_add_one(uint8_t* data, int length) {
    for (int i = 0; i < length; i++) {
        data[i] = (uint8_t)(data[i] + 1);
    }
}

/*
 * BGR -> RGB + crop + nearest-neighbor resize.
 *
 * Right now this is a plain C implementation.
 * Once you're comfortable, you can replace the inner math with FastCV calls.
 */
__declspec(dllexport)
void bgr_to_rgb_crop_resize_fastcv(
    const uint8_t* src, int src_w, int src_h,
    int crop_x, int crop_y, int crop_w, int crop_h,
    uint8_t* dst, int dst_w, int dst_h
) {
    for (int dy = 0; dy < dst_h; dy++) {
        int sy = crop_y + (dy * crop_h) / dst_h;
        if (sy < 0) sy = 0;
        if (sy >= src_h) sy = src_h - 1;

        for (int dx = 0; dx < dst_w; dx++) {
            int sx = crop_x + (dx * crop_w) / dst_w;
            if (sx < 0) sx = 0;
            if (sx >= src_w) sx = src_w - 1;

            int src_idx = (sy * src_w + sx) * 3;
            int dst_idx = (dy * dst_w + dx) * 3;

            uint8_t b = src[src_idx + 0];
            uint8_t g = src[src_idx + 1];
            uint8_t r = src[src_idx + 2];

            dst[dst_idx + 0] = r;
            dst[dst_idx + 1] = g;
            dst[dst_idx + 2] = b;
        }
    }
}

#ifdef __cplusplus
}
#endif