import cv2
from featurehub.fastcv_bridge import fastcv_crop_resize_bgr_to_rgb

img_bgr = cv2.imread("C:\\Users\\Arushi Taneja\\Downloads\\b2ap3_large_ee72093c-3c01-433a-8d25-701cca06c975.jpg")
h, w, _ = img_bgr.shape

rgb = fastcv_crop_resize_bgr_to_rgb(
    img_bgr,
    0, 0, w, h,
    224, 224
)

print(rgb.shape)  # (224, 224, 3)
cv2.imshow("out", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
