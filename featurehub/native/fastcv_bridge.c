// featurehub/native/fastcv_bridge.c
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// Just a test function: adds 1 to every byte in the buffer.
_declspec(dllexport)
void dummy_add_one(uint8_t* data, int length) {
    for (int i = 0; i < length; i++) {
        data[i] = (uint8_t)(data[i] + 1);
    }
}

#ifdef __cplusplus
}
#endif
