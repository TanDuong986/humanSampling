#!/home/dtan/anaconda3/envs/py37/bin/python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

import os
import datetime
import numpy as np

# Global variables
# loc = os.getcwd() # this is location of terminal
loc = os.path.dirname(os.path.abspath(__file__))
timee = datetime.datetime.now()
output_file = os.path.join(loc,f'video_{timee.date()}_{np.random.randint(1,1000)}.mp4')

fps = 30  # Frames per second
bridge = CvBridge()
video_writer = None

def image_callback(msg):
    global video_writer

    try:
        # Convert ROS Image message to OpenCV image
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Initialize video writer if it hasn't been initialized
        if video_writer is None:
            height, width, _ = cv_image.shape
            video_writer = cv2.VideoWriter(output_file,
                                           cv2.VideoWriter_fourcc(*'mp4v'),
                                           fps, (width, height))
            print(f' Created {output_file}')

        # Write frame to video file
        video_writer.write(cv_image)

        # Display the frame (optional)

        cv2.imshow('Frame', cv_image)
        cv2.waitKey(1)

    except Exception as e:
        rospy.logerr('Error processing image: %s', str(e))

def main():
    rospy.init_node('image_to_video_node', anonymous=False)
    rospy.Subscriber('/camera/color/image_raw', Image, image_callback)

    # Spin ROS event loop
    rospy.spin()

    # Release the video writer and close OpenCV windows
    if video_writer is not None:
        video_writer.release()
        print("\n Saved video")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    
