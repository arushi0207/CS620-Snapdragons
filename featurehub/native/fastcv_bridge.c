#include <stdint.h>
#include "fastcv.h"   // fine to keep even if we aren't calling into it yet

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float x;
    float y;
    float width;
    float height;
} FcvRect;

// Stub for now: no actual FastCV call yet
__declspec(dllexport)
int fastcv_detect_faces(
    const uint8_t* gray,
    int width,
    int height,
    FcvRect* out_rects,
    int max_faces
) {
    (void)gray;
    (void)width;
    (void)height;
    (void)out_rects;
    (void)max_faces;

    return 0;
}

#ifdef __cplusplus
}
#endif
