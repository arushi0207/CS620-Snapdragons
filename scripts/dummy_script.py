import numpy as np
from featurehub.extractors.mediapipe_face_extractor import MediaPipeFaceExtractor

extractor = MediaPipeFaceExtractor()
extractor.setup()

# Fake frame (BGR)
frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
out = extractor.extract(frame)

print(out.keys())
print(out["mediapipe_face"]["bbox"])
print(out["mediapipe_face"]["keypoints"])
