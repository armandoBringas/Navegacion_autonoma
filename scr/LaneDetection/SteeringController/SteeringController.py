import numpy as np
import cv2


class SteeringController:
    def __init__(self, kp=0.01):
        self.kp = kp  # Proportional control factor

    def calculate_steering_angle(self, lane_image):
        height, width, _ = lane_image.shape
        image_center = width // 2

        # Find the red pixels which represent the lanes
        lane_lines = cv2.findNonZero(cv2.inRange(lane_image, (255, 0, 0), (255, 0, 0)))
        if lane_lines is not None:
            lane_x_coords = lane_lines[:, 0][:, 0]
            lane_center = np.mean(lane_x_coords)
            deviation = image_center - lane_center
            return self.kp * deviation
        return 0
