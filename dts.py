#!/home/dtan/anaconda3/envs/py37/bin/python3

import cv2
import matplotlib.pyplot as plt
import time
import os
import numpy as np
import uuid
from pascal_voc_writer import Writer
from sub_node_image import ImageSubscriber
import rospy

def areaBox(box):
    x1,x2,x3,x4 = box 
    return (x3-x1)*(x4-x2)

def iou(box1,box2,thresh):
    x1_n = max(box1[0],box2[0])
    y1_n = max(box1[1],box2[1])
    x2_n = min(box1[2],box2[2])
    y2_n = min(box1scale = 3)
    font = ([3],box2[3])
    w = max(x2_n-x1_n,0)
    h = max(y2_n-y1_n,0)
    area = w*h
    if area == 0:
        return 0
    else:
        intersection = areaBox(box1)+ areaBox(box2) - area
        iou = area/intersection
        return iou>= thresh
    

if __name__ =="__main__":
    loc = os.getcwd()
    config_file = '/home/dtan/catkin_ws/src/transfer_img/scripts/humanSampling/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    frozen_model = '/home/dtan/catkin_ws/src/transfer_img/scripts/humanSampling/frozen_inference_graph.pb'
    model = cv2.dnn_DetectionModel(frozen_model,config_file)

    model.setInputSize(320,320)
    model.setInputScale(1.0/127.5) # 255 / 2 = 127.5
    model.setInputMean((127.5, 127.5, 127.5)) # mobilenet => [-1, 1]
    model.setInputSwapRB(True)

    output_path = {"img":"/home/dtan/catkin_ws/src/transfer_img/scripts/humanSampling/data/img","label":"/home/dtan/catkin_ws/src/transfer_img/scripts/humanSampling/data/label"}
    for p in output_path.values():
        if not os.path.exists(p):
            os.makedirs(p)

    classlabels = []
    file_name = '/home/dtan/catkin_ws/src/transfer_img/scripts/humanSampling/labels.txt'
    with open(file_name, 'rt') as fpt:
        classlabels = fpt.read().rstrip('\n').split('\n')

    prev_time =0
    new_time = 0
    runOutTime = 3

    font_scale = 3
    font = cv2.FONT_HERSHEY_PLAIN

    take_size_id = 0
    hwc = [0,0,0]
    try:
        image_subscriber = ImageSubscriber()
        while not rospy.is_shutdown():
            processed_image = image_subscriber.return_image()
            if processed_image is not None:
                if take_size_id ==0:
                    hwc[:] = processed_image.shape
                # Do something with the processed image
                ClassIndex, conf,bbox = model.detect(processed_image,confThreshold=0.6)
                if (1 in ClassIndex) :
                    img = processed_image.copy()
                    new_time = time.time()
                    idd = str(uuid.uuid1())
                    writer = Writer(f'/home/dtan/catkin_ws/src/transfer_img/scripts/humanSampling/data/label/image_{idd}.jpg', hwc[1],hwc[0])
                    for ClassInd, conf, boxes in zip(ClassIndex.flatten(),conf.flatten(),bbox):
                        if ClassInd == 1 :
                            (x1,x2,x3,x4) = boxes #x3,x4 is width and height
                            cv2.rectangle(processed_image,boxes,(255,0,0),2) # this use type of x,y,w,h but when write, use type xmin, ymin, xmax, ymax
                            cv2.putText(processed_image,classlabels[ClassInd-1],(boxes[0],boxes[1]), font,fontScale=font_scale,color=(0,255,0))
                            # cv2.putText(frame,f'p1({x1},{x2}) | p2({x3},{x4})',(boxes[0],boxes[1]), font,fontScale=font_scale,color=(0,255,0))
                            
                            # add objects (class, xmin, ymin, xmax, ymax)
                            writer.addObject(classlabels[ClassInd-1], x1,x2,x3+x1,x4+x2)
                    if new_time - prev_time >= runOutTime:
                        cv2.imwrite(f'/home/dtan/catkin_ws/src/transfer_img/scripts/humanSampling/data/img/image_{idd}.jpg',img)
                        writer.save(f'/home/dtan/catkin_ws/src/transfer_img/scripts/humanSampling/data/label/image_{idd}.xml')
                        print(f'taken {uuid}')
                        prev_time = new_time
                cv2.imshow("OB",processed_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("No signal")

    except rospy.ROSInterruptException:
        pass
