import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread('/home/dtan/catkin_ws/src/transfer_img/scripts/humanSampling/data/img/image_b4eb37e1-f8c8-11ed-8441-5cbaef0897af.jpg')
cv2.rectangle(image,[369,239,496,519],(255,0,0),3)

plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.show()