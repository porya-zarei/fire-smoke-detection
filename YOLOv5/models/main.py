from PIL import Image
import cv2
from FireSmokeDetector_YOLOv5 import FireSmokeDetector_YOLOv5
img_paths = [
    "../../data/Fire dataset.v1i.yolov8/valid/images/0051_jpg.rf.683c49c744e3c21af25e725db73f6f44.jpg",
    "../../data/Fire dataset.v1i.yolov8/valid/images/0055_jpg.rf.194a5c23fc80f472eca8d0e8c285d25a.jpg",
    "../../data/Fire dataset.v1i.yolov8/valid/images/0066_jpg.rf.cc7a68ea547bd2f507f00980c12123dd.jpg",
    "../../data/Fire dataset.v1i.yolov8/valid/images/0143_jpg.rf.e7948430599f8f056a70b5e4e6fa717b.jpg",
    "../../data/Fire dataset.v1i.yolov8/valid/images/0198_jpg.rf.76c89f7dec12c1507382a81cd393c9b7.jpg",
    "../../data/Fire dataset.v1i.yolov8/valid/images/1861bf1c2bbb6cfd_jpg.rf.93752bdd697e2a69b8bd6d3ee094a8bc.jpg",
    "../../data/Fire dataset.v1i.yolov8/valid/images/772_jpg.rf.b6214c0bf66258ac425952c297976133.jpg"
]
imgs = [Image.open(i).convert("RGB") for i in img_paths]

# video = cv2.VideoCapture("../../../top-tree-detection/src/data/testing/video.mp4")

class_names = ["Fire","Fire"]
# Step 1: Initialize model with the best available weights
# YOLOS model
weights_path = "./best-v5-1.pt"

predictor = FireSmokeDetector_YOLOv5(weights_path,class_names,threshold=0.1)

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
