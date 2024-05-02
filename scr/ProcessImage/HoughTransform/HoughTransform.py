import cv2
import numpy as np


class HoughTransform:
    def __init__(self, image):
        self.image = image
        self.img_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.img_blur = cv2.GaussianBlur(self.img_gray, (5, 5), 0)
        self.img_canny = cv2.Canny(self.img_blur, 40, 120)
        self.rho = 2
        self.theta = np.pi / 180
        self.threshold = 40
        self.min_line_length = 50
        self.max_line_gap = 10
        self.alpha = 1
        self.beta = 1
        self.gamma = 1

    @staticmethod
    def greyscale_cv2(image):
        """
        Processing image to greyscale
        :param image:
        :return:
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def img_roi_mask(self):
        vertices = np.array([[(20, 540), (200, 300), (650, 300), (850, 540)]], dtype=np.int32)
        img_roi = np.zeros_like(self.img_gray)
        cv2.fillPoly(img_roi, vertices, 255)
        return cv2.bitwise_and(self.img_canny, img_roi)

    def img_lane_lines(self):
        lines = cv2.HoughLinesP(self.img_roi_mask(),
                                self.rho, self.theta,
                                self.threshold, np.array([]),
                                self.min_line_length,
                                self.max_line_gap)

        img_lines = np.zeros((self.img_roi_mask.shape[0], self.img_roi_mask.shape[1], 3), dtype=np.uint8)

        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img_lines, (x1, y1), (x2, y2), [255, 0, 0], 30)

        print(lines)

        return cv2.addWeighted(self.img_roi_mask(), self.alpha, img_lines, 1 - self.alpha, 0, 0)
