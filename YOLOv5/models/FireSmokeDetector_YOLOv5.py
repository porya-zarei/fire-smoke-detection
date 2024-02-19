import numpy as np
import torch
import cv2
from time import time


class FireSmokeDetector_YOLOv5:
    def __init__(self,weights_path,class_names=[],threshold=0.5):
        self.fps = 0
        self.class_names = class_names
        self.model = self.__load_model(weights_path)
        self.finded = False

    def __load_model(self,weights_path):
        model = torch.hub.load('ultralytics/yolov5','custom',weights_path,skip_validation=True,force_reload=False)
        return model

    def __get_label(self,classes,box_data):
        index = int(box_data[5])
        return classes[index]

    def __draw_box(self,img,box_data,color=(0,255,00),thickness=2):
        xmin = int(box_data[0])
        ymin = int(box_data[1])
        xmax = int(box_data[2])
        ymax = int(box_data[3])
        label = self.__get_label(self.class_names,box_data)
        score = box_data[4]
        # print(f"{label}:{score},xmin:{xmin},ymin:{ymin},xmax:{xmax},ymax:{ymax}")
        cv2.rectangle(img,(xmin,ymin),(xmax,ymax),color,thickness)
        cv2.putText(img,f"{label}({score:0.3f})",(xmin,ymin-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
        return img

    def __draw_boxes(self,img,box_data,color=(0,255,00),thickness=2):
        self.finded = False
        for box in box_data:
            img = self.__draw_box(img,box,color,thickness)
        if len(box_data) > 0:
            self.finded = True
        cv2.putText(img,f"{self.fps:0.3f} fps",(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(200,0,0),2)
        return img

    def predict(self,img):
        """
            [[1.92891e+02, 3.31313e+02, 2.20479e+02, 3.57784e+02, 3.97579e-01, 0.00000e+00],
            [1.32464e+02, 3.26052e+02, 1.65118e+02, 3.55398e+02, 3.18138e-01, 0.00000e+00],
            [2.81917e+02, 3.12348e+02, 3.06335e+02, 3.34977e+02, 2.70851e-01, 0.00000e+00],
            [3.69030e+02, 2.11246e+02, 3.89584e+02, 2.32369e+02, 2.52894e-01, 0.00000e+00]]
        -------------------------------------------------------------------------------
                 xmin        ymin        xmax        ymax  confidence  class      name
            0  192.891220  331.313446  220.478500  357.783813    0.397579      0  tree-top
            1  132.464310  326.052490  165.118362  355.398132    0.318138      0  tree-top
            2  281.917450  312.347931  306.334717  334.977081    0.270851      0  tree-top
            3  369.029907  211.246231  389.583618  232.368851    0.252894      0  tree-top
        """
        start_time = time()
        self.img = img
        self.prediction = self.model(self.img).xyxy[0]
        end_time = time()
        pred_time = end_time - start_time
        self.fps = 1.0/pred_time
        return self.prediction

    def draw(self,img=None,color=(0,255,00),thickness=2):
        if img is not None:
            self.img = img
        self.img = np.array(img)
        self.img = self.img[:,:,::-1].copy()
        self.img = self.__draw_boxes(self.img,self.prediction,color,thickness)
        # self.img = cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
        return self.img

    def is_finded(self):
        return self.finded
