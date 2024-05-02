from controller import Display
import numpy as np
import cv2


class SimpleController:
    def __init__(self, angle, speed, steering_angle, manual_steering):
        """
        :param angle:
        :param speed:
        """
        self.angle = angle
        self.speed = speed
        self.steering_angle = steering_angle
        self.manual_steering = manual_steering

    @staticmethod
    def get_image_from_camera(camera):
        """
        Getting image from camera
        :param camera:
        :return:
        """
        raw_image = camera.getImage()
        return np.frombuffer(raw_image, np.uint8).reshape(
            (camera.getHeight(), camera.getWidth(), 4))

    @staticmethod
    def greyscale_cv2(image):
        """
        Processing image to greyscale
        :param image:
        :return:
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def display_image(display, image):
        # Image to display
        image_rgb = np.dstack((image, image, image,))
        # Display image
        image_ref = display.imageNew(
            image_rgb.tobytes(),
            Display.RGB,
            width=image_rgb.shape[1],
            height=image_rgb.shape[0],
        )
        display.imagePaste(image_ref, 0, 0, False)

    def set_steering_angle(self, wheel_angle):
        # Check limits of steering
        if (wheel_angle - self.steering_angle) > 0.1:
            wheel_angle = self.steering_angle + 0.1
        if (wheel_angle - self.steering_angle) < -0.1:
            wheel_angle = self.steering_angle - 0.1
        self.steering_angle = wheel_angle

        # limit range of the steering angle
        if wheel_angle > 0.5:
            wheel_angle = 0.5
        elif wheel_angle < -0.5:
            wheel_angle = -0.5

        # update steering angle
        self.angle = wheel_angle

    def change_steer_angle(self, inc):
        # Apply increment
        new_manual_steering = self.manual_steering + inc

        # Validate interval
        if 25.0 >= new_manual_steering >= -25.0:
            self.manual_steering = new_manual_steering
            self.set_steering_angle(self.manual_steering * 0.02)
        # Debugging
        if self.manual_steering == 0:
            print("going straight")
        else:
            turn = "left" if self.steering_angle < 0 else "right"
            print("turning {} rad {}".format(str(self.steering_angle), turn))
