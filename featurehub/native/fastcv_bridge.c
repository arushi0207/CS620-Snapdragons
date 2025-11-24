#include <stdint.h>
#include <stdlib.h>
#include "fastcv.h"

#ifdef __cplusplus
extern "C" {
#endif

// This must match the Python FcvRect struct
typedef struct
{
    int32_t  x;
    int32_t  y;
    uint32_t width;
    uint32_t height;
} FcvRect;

// Fallback: center box heuristic
static int make_center_box(int width, int height, FcvRect* out_rects)
{
    if (!out_rects || width <= 0 || height <= 0) {
        return 0;
    }

    int min_side = (width < height) ? width : height;
    int box_side = (int)(min_side * 0.6f);
    if (box_side <= 0) {
        return 0;
    }
    if (box_side > width)  box_side = width;
    if (box_side > height) box_side = height;

    int x = (width  - box_side) / 2;
    int y = (height - box_side) / 2;
    if (x < 0) x = 0;
    if (y < 0) y = 0;

    out_rects[0].x      = x;
    out_rects[0].y      = y;
    out_rects[0].width  = (uint32_t)box_side;
    out_rects[0].height = (uint32_t)box_side;

    return 1;
}

// -----------------------------
// FAST + Hough-circle detector
// -----------------------------
__declspec(dllexport)
int fastcv_detect_faces(
    const uint8_t* gray,  // grayscale image, HxW
    int width,
    int height,
    FcvRect* out_rects,   // output array
    int max_faces
) {
    if (!gray || !out_rects || max_faces <= 0 || width <= 0 || height <= 0) {
        return 0;
    }

    const uint32_t uwidth  = (uint32_t)width;
    const uint32_t uheight = (uint32_t)height;
    const uint32_t srcStride = uwidth;  // 1 byte per pixel

    // --------------------
    // 1) Optional: FAST corners (exercise FASTCV FAST API)
    // --------------------
    {
        const uint32_t nCornersMax = 4096;  // cap
        uint32_t* corners_xy = (uint32_t*)malloc(sizeof(uint32_t) * 2 * nCornersMax);
        uint32_t nCorners = 0;

        if (corners_xy) {
            // barrier ~20, border = 3 is typical
            fcvCornerFast10u8(
                gray,
                uwidth,
                uheight,
                srcStride,
                20,        // barrier
                3,         // border
                corners_xy,
                nCornersMax,
                &nCorners
            );
            free(corners_xy);
        }
    }

    // --------------------
    // 2) Build a binary image for HoughCircle
    // --------------------
    uint8_t* bin = (uint8_t*)malloc(width * height);
    if (!bin) {
        return make_center_box(width, height, out_rects);
    }

    const int threshold = 90;  // simple global threshold
    for (int y = 0; y < height; ++y) {
        const uint8_t* row_in  = gray + y * width;
        uint8_t*       row_out = bin  + y * width;
        for (int x = 0; x < width; ++x) {
            row_out[x] = (row_in[x] > threshold) ? 255 : 0;
        }
    }

    // --------------------
    // 3) Hough circle detection using FastCV
    // --------------------

    // Circles buffer
    const uint32_t maxCircle = (max_faces > 0) ? (uint32_t)max_faces : 1u;
    fcvCircle* circles = (fcvCircle*)malloc(sizeof(fcvCircle) * maxCircle);
    if (!circles) {
        free(bin);
        return make_center_box(width, height, out_rects);
    }

    uint32_t numCircle = 0;

    // Scratch buffer recommended size: 16 * srcStride * srcHeight bytes
    uint64_t scratchBytes64 = 16ULL * (uint64_t)srcStride * (uint64_t)uheight;
    if (scratchBytes64 > 0x7FFFFFFFULL) {
        scratchBytes64 = 0x7FFFFFFFULL;   // clamp to avoid silly sizes
    }
    uint32_t scratchBytes = (uint32_t)scratchBytes64;
    void* scratch = malloc(scratchBytes);
    if (!scratch) {
        free(circles);
        free(bin);
        return make_center_box(width, height, out_rects);
    }

    // Heuristic parameters
    const uint32_t minSide = (uwidth < uheight) ? uwidth : uheight;
    const uint32_t minRadius = (uint32_t)(0.10f * (float)minSide);
    const uint32_t maxRadius = (uint32_t)(0.50f * (float)minSide);
    const uint32_t minDist   = (uint32_t)(0.25f * (float)minSide);

    const uint32_t cannyThreshold = 100;
    const uint32_t accThreshold   = 30;

    fcvHoughCircleu8(
        bin,
        uwidth,
        uheight,
        srcStride,
        circles,
        &numCircle,
        maxCircle,
        minDist,
        cannyThreshold,
        accThreshold,
        minRadius,
        maxRadius,
        scratch
    );

    free(scratch);
    free(bin);

    if (numCircle == 0) {
        free(circles);
        return make_center_box(width, height, out_rects);
    }

    // --------------------
    // 4) Pick the best circle and convert to bbox
    // --------------------
    int bestIdx = 0;
    int bestArea = -1;

    for (uint32_t i = 0; i < numCircle && i < maxCircle; ++i) {
        int r = circles[i].radius;
        if (r <= 0) continue;
        int area = r * r;
        if (area > bestArea) {
            bestArea = area;
            bestIdx = (int)i;
        }
    }

    if (bestArea <= 0) {
        free(circles);
        return make_center_box(width, height, out_rects);
    }

    fcvCircle best = circles[bestIdx];
    free(circles);

    int cx = best.x;
    int cy = best.y;
    int r  = best.radius;

    // Enlarge horizontally & vertically to cover more of the face
    // and shift the box slightly up (faces extend more downward than upward
    // from the eye region).
    float scale_x = 1.6f;   // widen box (was 1.0)
    float scale_y = 2.0f;   // taller box (was 1.0)
    float shift_up = 0.3f;  // shift box up by 0.3 * radius

    int half_w = (int)(r * scale_x);
    int half_h = (int)(r * scale_y);

    // Shift the center up a bit so chin fits in the box
    int cx_shifted = cx;
    int cy_shifted = cy - (int)(shift_up * (float)r);

    int x0 = cx_shifted - half_w;
    int y0 = cy_shifted - half_h;
    int w  = 2 * half_w;
    int h  = 2 * half_h;

    // Clamp to image bounds
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x0 + w > width)  w = width  - x0;
    if (y0 + h > height) h = height - y0;

    if (w <= 0 || h <= 0) {
        return make_center_box(width, height, out_rects);
    }

    out_rects[0].x      = x0;
    out_rects[0].y      = y0;
    out_rects[0].width  = (uint32_t)w;
    out_rects[0].height = (uint32_t)h;

    return 1;

}

#ifdef __cplusplus
}
#endif
