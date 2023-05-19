#!/home/dtan/anaconda3/envs/py37/bin/python3

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class ImageSubscriber:
    def __init__(self):
        rospy.init_node('image_subscriber_node', anonymous=True)
        self.bridge = CvBridge()
        self.subscriber = rospy.Subscriber('/image_topic', Image, self.callback)
        self.cv_image = None

    def callback(self, image_msg):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            self.return_image()
        except Exception as e:
            rospy.logerr(e)
            return

    def return_image(self):
        if self.cv_image is None:
            # rospy.logerr('No image received yet')
            # return None
            pass

        # Do some processing on the image (optional)
        processed_image = self.cv_image

        return processed_image

if __name__ == '__main__':
    try:
        image_subscriber = ImageSubscriber()
        while not rospy.is_shutdown():
            processed_image = image_subscriber.return_image()
            if processed_image is not None:
                # Do something with the processed image
                cv2.imshow('Image',processed_image)
                cv2.waitKey(1)
    except rospy.ROSInterruptException:
        pass