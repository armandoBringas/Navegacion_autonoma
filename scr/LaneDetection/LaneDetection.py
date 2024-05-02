import cv2


class LaneDetection:
    @staticmethod
    def greyscale_cv2(image):
        """
        Processing image to greyscale
        :param image:
        :return:
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
