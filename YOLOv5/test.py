from ultralytics import YOLO
from PIL import Image

# Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('./models/best.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
image = Image.open("../data/xml/valid/2018_SJER_3_254000_4109000_image_337_jpeg.rf.1962417cafa67a2bcb3c5f8a5092465e.jpg")
# Train the model
result = model.predict(image)[0]

print(f"result => {result}")

print(f"boxes => {result.boxes}")

# draw boxes
for i,box in enumerate(result.boxes):
    print(f"box => {i}-{box}")