from PIL import Image
import cv2
from TreeTopViewDetector_YOLOv5 import TreeTopViewDetector_YOLOv5
img_paths = [
    "../../../top-tree-detection/src/data/testing/3.jpg",
    "../../../top-tree-detection/src/data/testing/4.jpg",
    "../../../top-tree-detection/src/data/testing/5.jpg",
    "../../../top-tree-detection/src/data/testing/6.jpg",
    "../../../top-tree-detection/src/data/testing/7.jpg",
    "../../../top-tree-detection/src/data/testing/ABBY_065_2019_2.jpeg",
    "../../../top-tree-detection/src/data/testing/ABBY_029_2019.jpeg"
]
imgs = [Image.open(i).convert("RGB") for i in img_paths]

# video = cv2.VideoCapture("../../../top-tree-detection/src/data/testing/video.mp4")

class_names = ["tree-top"]
# Step 1: Initialize model with the best available weights
# YOLOS model
weights_path = "./best.pt"

predictor = TreeTopViewDetector_YOLOv5(weights_path,class_names,threshold=0.8)

for i in range(len(imgs)):
    img = imgs[i]
    predictor.predict(img)
    img = predictor.draw(img)
    cv2.imshow(f"img-{i}",img)

# while True:
#     ret, frame = video.read()
#     if not ret:
#         break
#     img = Image.fromarray(frame)
#     predictor.predict(img)
#     img = predictor.draw(img)
#     cv2.imshow("img",img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

cv2.waitKey(0)
