#include <stdint.h>
#include <stdlib.h>
#include "fastcv.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    int32_t  x;
    int32_t  y;
    uint32_t width;
    uint32_t height;
} FcvRect;

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

__declspec(dllexport)
int fastcv_detect_faces(
    const uint8_t* gray,
    int width,
    int height,
    FcvRect* out_rects,
    int max_faces
) {
    if (!gray || !out_rects || max_faces <= 0 || width <= 0 || height <= 0) {
        return 0;
    }

    // FastCV requirement from docs: width > 50, height > 5
    if (width <= 50 || height <= 5) {
        return make_center_box(width, height, out_rects);
    }

    const unsigned int uwidth  = (unsigned int)width;
    const unsigned int uheight = (unsigned int)height;

    // -------- 1. Init MSER ----------
    void* mserHandle = NULL;

    unsigned int delta       = 2;
    unsigned int minArea     = 30;
    unsigned int maxArea     = (unsigned int)(0.25f * (float)uwidth * (float)uheight);
    if (maxArea < minArea) {
        maxArea = minArea;
    }
    float maxVariation  = 0.15f;
    float minDiversity  = 0.20f;

    int ok = fcvMserInit(
        uwidth,
        uheight,
        delta,
        minArea,
        maxArea,
        maxVariation,
        minDiversity,
        &mserHandle
    );

    if (!ok || !mserHandle) {
        return make_center_box(width, height, out_rects);
    }

    // -------- 2. Allocate output buffers correctly ----------

    // Use a generous maxContours capacity, independent of max_faces.
    unsigned int maxContours = 256;

    unsigned int* numPointsInContour =
        (unsigned int*)malloc(sizeof(unsigned int) * maxContours);
    if (!numPointsInContour) {
        fcvMserRelease(mserHandle);
        return make_center_box(width, height, out_rects);
    }

    // Docs say: typical size = (#pixels) * 30
    unsigned long long totalPixels = (unsigned long long)uwidth * (unsigned long long)uheight;
    unsigned long long allocCount  = totalPixels * 30ULL;
    if (allocCount > 1000000000ULL) {  // clamp ~1e9 to avoid insane allocations
        allocCount = 1000000000ULL;
    }
    unsigned int pointsArraySize = (unsigned int)allocCount;

    unsigned int* pointsArray =
        (unsigned int*)malloc(sizeof(unsigned int) * pointsArraySize);
    if (!pointsArray) {
        free(numPointsInContour);
        fcvMserRelease(mserHandle);
        return make_center_box(width, height, out_rects);
    }

    unsigned int numContours = 0;

    // -------- 3. Run MSER ----------
    fcvMseru8(
        mserHandle,
        gray,
        uwidth,
        uheight,
        uwidth,             // srcStride
        maxContours,
        &numContours,
        numPointsInContour,
        pointsArraySize,
        pointsArray
    );

    fcvMserRelease(mserHandle);

    if (numContours == 0) {
        free(pointsArray);
        free(numPointsInContour);
        return make_center_box(width, height, out_rects);
    }

    // -------- 4. Pick "best" contour (area & closeness to center) ----------

    float img_cx = 0.5f * (float)(width  - 1);
    float img_cy = 0.5f * (float)(height - 1);

    int   best_idx   = -1;
    float best_score = -1.0f;

    for (unsigned int c = 0; c < numContours && c < maxContours; ++c) {
        unsigned int start = (c == 0) ? 0u : numPointsInContour[c - 1];
        unsigned int end   = numPointsInContour[c];

        if (end <= start || end > pointsArraySize) {
            continue;
        }

        int minX = width - 1;
        int maxX = 0;
        int minY = height - 1;
        int maxY = 0;

        for (unsigned int i = start; i < end; ++i) {
            unsigned int idx = pointsArray[i];
            unsigned int py  = idx / uwidth;
            unsigned int px  = idx % uwidth;

            if (px >= (unsigned int)width || py >= (unsigned int)height) {
                continue;
            }

            int x = (int)px;
            int y = (int)py;

            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
        }

        if (minX > maxX || minY > maxY) {
            continue;
        }

        int box_w = maxX - minX + 1;
        int box_h = maxY - minY + 1;
        if (box_w <= 0 || box_h <= 0) {
            continue;
        }

        // reject huge regions relative to maxArea
        int area = box_w * box_h;
        if (area > (int)maxArea) {
            continue;
        }

        float cx = 0.5f * (float)(minX + maxX);
        float cy = 0.5f * (float)(minY + maxY);

        float dx = cx - img_cx;
        float dy = cy - img_cy;
        float dist2 = dx * dx + dy * dy;

        // score: big + central
        float score = (float)area / (1.0f + 0.001f * dist2);
        if (score > best_score) {
            best_score = score;
            best_idx   = (int)c;
        }
    }

    int num_out = 0;

    if (best_idx >= 0) {
        unsigned int start = (best_idx == 0) ? 0u : numPointsInContour[best_idx - 1];
        unsigned int end   = numPointsInContour[best_idx];

        if (end > start && end <= pointsArraySize) {
            int minX = width - 1;
            int maxX = 0;
            int minY = height - 1;
            int maxY = 0;

            for (unsigned int i = start; i < end; ++i) {
                unsigned int idx = pointsArray[i];
                unsigned int py  = idx / uwidth;
                unsigned int px  = idx % uwidth;

                if (px >= (unsigned int)width || py >= (unsigned int)height) {
                    continue;
                }

                int x = (int)px;
                int y = (int)py;

                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }

            if (minX <= maxX && minY <= maxY) {
                out_rects[0].x      = minX;
                out_rects[0].y      = minY;
                out_rects[0].width  = (uint32_t)(maxX - minX + 1);
                out_rects[0].height = (uint32_t)(maxY - minY + 1);
                num_out = 1;
            }
        }
    }

    free(pointsArray);
    free(numPointsInContour);

    if (num_out == 0) {
        return make_center_box(width, height, out_rects);
    }

    // We only ever output 1 face for now, regardless of max_faces.
    return num_out;
}

#ifdef __cplusplus
}
#endif
