import cv2
import numpy as np
from featurehub.fastcv_bridge import fastcv_detect_faces

img_bgr = cv2.imread("C:\\Users\\Arushi Taneja\\Downloads\\b2ap3_large_ee72093c-3c01-433a-8d25-701cca06c975.jpg")
h, w, _ = img_bgr.shape

img = np.random.randint(0, 255, (240, 320), dtype=np.uint8)

rects = fastcv_detect_faces(img)
print("Detected faces:", rects)