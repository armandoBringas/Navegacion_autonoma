import cv2
import numpy as np


class HoughTransform:
    def __init__(self, image, vertices, flip: bool = False):
        # Configuration parameters
        self.rho = 1
        self.theta = np.pi / 180
        self.threshold = 25
        self.min_line_length = 10
        self.max_line_gap = 2
        self.alpha = 1
        self.beta = 1
        self.gamma = 0  # Typically set to zero unless you have a specific need for a gamma correction

        self.image = image
        self.vertices = vertices
        self.flip = flip

        # Process image
        self.img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        if self.flip:
            self.img_rgb = cv2.flip(self.img_rgb, 0)  # Flip vertically

        self.img_gray = cv2.cvtColor(self.img_rgb, cv2.COLOR_RGB2GRAY)
        self.img_blur = cv2.GaussianBlur(self.img_gray, (7, 7), 0)
        self.img_canny = cv2.Canny(self.img_blur, 75, 200)

    def img_roi_mask(self):
        mask = np.zeros(self.img_rgb.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (0, 0), (mask.shape[1], int(mask.shape[0] * 0.5)), 255, thickness=-1)
        return cv2.bitwise_and(self.img_canny, mask)

    def detect_lanes(self):
        img_roi_mask = self.img_roi_mask()
        lines = cv2.HoughLinesP(img_roi_mask,
                                self.rho, self.theta,
                                self.threshold, np.array([]),
                                self.min_line_length,
                                self.max_line_gap)

        line_image = np.zeros_like(self.img_rgb)
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Combine the line image with the original image
        combined_image = cv2.addWeighted(self.img_rgb, self.alpha, line_image, self.beta, self.gamma)

        return combined_image


