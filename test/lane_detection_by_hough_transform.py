import scr.ProcessImage.HoughTransform.HoughTransform as HoughTransform
import matplotlib.pyplot as plt
import cv2

img_path = '../img/image_lane_c.jpg'
img_bgr = cv2.imread(img_path)
img_lane_lines = HoughTransform.HoughTransform(img_bgr).img_lane_lines()

plt.figure()
plt.imshow(img_lane_lines)
plt.show()
