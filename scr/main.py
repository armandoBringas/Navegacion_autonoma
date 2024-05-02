from controller import Display, Keyboard
from vehicle import Car, Driver
from SimpleController.SimpleController import SimpleController
from ProcessImage.HoughTransform.HoughTransform import HoughTransform
import cv2
from datetime import datetime
import os


def main():
    # Set initial parameters
    manual_steering = 0
    steering_angle = 0
    angle = 0.0
    speed = 30

    # Create the Robot instance
    robot = Car()
    driver = Driver()
    simple_controller = SimpleController(angle, speed, steering_angle, manual_steering)

    # Get the time step of the current world
    time_step = int(robot.getBasicTimeStep())

    # Create the camera instance
    camera = robot.getDevice('camera')
    camera.enable(time_step)

    # Processing display
    displayed_img = Display('display_image')

    # Create keyboard instance
    keyboard = Keyboard()
    keyboard.enable(time_step)

    while robot.step() != -1:
        # Get image from camera
        image = SimpleController.get_image_from_camera(camera)

        # Process and display the image
        grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        simple_controller.display_image(displayed_img, grey_image)

        # Read keyboard
        key = keyboard.getKey()

        if key == keyboard.UP:  # up
            simple_controller.speed = speed + 5.0
            print("up")
        elif key == keyboard.DOWN:  # down
            simple_controller.speed = speed - 5.0
            print("down")
        elif key == keyboard.RIGHT:  # right
            simple_controller.change_steer_angle(+1)
            print("right")
        elif key == keyboard.LEFT:  # left
            simple_controller.change_steer_angle(-1)
            print("left")
        elif key == ord('A'):
            # filename with timestamp and saved in current directory
            current_datetime = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
            file_name = current_datetime + ".png"
            print("Image taken")
            camera.saveImage(os.getcwd() + "/" + file_name, 1)

        # update angle and speed
        driver.setSteeringAngle(simple_controller.angle)
        driver.setCruisingSpeed(simple_controller.speed)


if __name__ == '__main__':
    main()
