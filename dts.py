import cv2
import matplotlib.pyplot as plt
import time
import os
import numpy as np

def areaBox(box):
    x1,x2,x3,x4 = box 
    return (x3-x1)*(x4-x2)

def iou(box1,box2,thresh):
    x1_n = max(box1[0],box2[0])
    y1_n = max(box1[1],box2[1])
    x2_n = min(box1[2],box2[2])
    y2_n = min(box1[3],box2[3])
    w = max(x2_n-x1_n,0)
    h = max(y2_n-y1_n,0)
    area = w*h
    if area == 0:
        return 0
    else:
        intersection = areaBox(box1)+ areaBox(box2) - area
        iou = area/intersection
        return iou>= thresh
# def sort(boxes,conf,thresh_conf=0.6,thresh_area=500):
#     mask1 = conf >= thresh_conf
#     for box in boxes:
#         mask2 = areaBox(box) >= thresh_area
#     mask = mask1*mask2
#     return mask

# boxes = np.array([[0,0,200,200],[50,50,100,100],[20,20,300,300]])
# confs = np.array([0.54,0.34,0.55])

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model,config_file)

model.setInputSize(320,320)
model.setInputScale(1.0/127.5) # 255 / 2 = 127.5
model.setInputMean((127.5, 127.5, 127.5)) # mobilenet => [-1, 1]
model.setInputSwapRB(True)

classlabels = []
file_name = 'labels.txt'
with open(file_name, 'rt') as fpt:
    classlabels = fpt.read().rstrip('\n').split('\n')

prev_time =0
new_time = 0

cap = cv2.VideoCapture(0)
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    
    ret,frame = cap.read()
    if ret:
        new_time = time.time()
        ClassIndex, conf,bbox = model.detect(frame,confThreshold=0.55)
        fps = 1/(new_time - prev_time)
        prev_time = new_time
        print(bbox)
        if len(ClassIndex) != 0 :
            for ClassInd, conf, boxes in zip(ClassIndex.flatten(),conf.flatten(),bbox):
                if ClassInd == 1 and conf >= 0.6 :
                    cv2.rectangle(frame,boxes,(255,0,0),2)
                    print(areaBox(boxes))
                    cv2.putText(frame,classlabels[ClassInd-1],(boxes[0]+10,boxes[1]+40), font,fontScale=font_scale,color=(0,255,0))
    
    cv2.imshow("OB",frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
