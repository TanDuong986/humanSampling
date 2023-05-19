# Human detection lightweight code throught OpenCV

**Acknowledgement:** This repo is an inherited from (this amazing)[https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection] repo. Thanks

**Make sure right the path of config, frozen and label file**

I write this code suitable for all ROS1 and normal run. (dst_new)[/dst_new.py] is use for normal run, It will read your camera on laptop and if image contains human, every 2 seconds it will save image and its label follow VOC formal in `.xml` file all in (data)[/data]

Second, for ROS, (dts)[/dts.py] will subscribe to a topic contains image which had configured in (sub_node)[/sub_node_image.py], everytime this topic receive image and have human, it will do the same. I do this for semi-sampling image human use for my human detection custom project.

