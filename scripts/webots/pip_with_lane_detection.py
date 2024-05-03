"""camera_pid controller with lane detection and dynamic control."""

from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2

# Constants for lane detection and control
CONTROL_COEFFICIENT = 0.002
SHOW_IMAGE_WINDOW = False

# Initial angle and speed settings
manual_steering = 0
steering_angle = 0
angle = 0.0
speed = 30  # Set initial cruising speed to 30 km/h

def get_image(camera):
    raw_image = camera.getImage()
    if raw_image is None:
        print("Camera image not available.")
        return None
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image

def greyscale_cv2(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_img

def display_image(display, image):
    image_rgb = np.dstack((image, image, image,))
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)

def set_speed(driver, kmh):
    global speed
    speed = kmh
    driver.setCruisingSpeed(speed)

def set_steering_angle(driver, wheel_angle):
    global angle, steering_angle
    # Implement steering control logic if needed
    angle = wheel_angle  # Update the global angle
    driver.setSteeringAngle(angle)

def regulate(driver, camera):
    global last_valid_steering, lost_frame_count
    img = get_image(camera)
    if img is None or img.size == 0:
        print("No image data to process.")
        return

    y_start = max(0, min(img.shape[0], 10))
    y_end = max(0, min(img.shape[0], 54))
    img_slice = img[y_start:y_end, :]

    if img_slice.size == 0:
        print("Empty image slice.")
        return

    try:
        img_hsv = cv2.cvtColor(img_slice, cv2.COLOR_RGB2HSV)
    except cv2.error as e:
        print(f"Error converting image to HSV: {e}")
        return

    mask = cv2.inRange(img_hsv, np.array([70, 120, 170]), np.array([120, 160, 210]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return

    largest_contour = max(contours, key=cv2.contourArea)
    if largest_contour.size == 0 or cv2.contourArea(largest_contour) == 0:
        return

    center = cv2.moments(largest_contour)
    if center['m00'] == 0:
        return

    center_x = int(center['m10'] / center['m00'])
    error = center_x - (img_slice.shape[1] / 2)
    new_steering = error * CONTROL_COEFFICIENT
    driver.setSteeringAngle(new_steering)

    if SHOW_IMAGE_WINDOW:
        cv2.imshow('Lane Detection', img_slice)
        cv2.waitKey(1)

def main():
    robot = Car()
    driver = Driver()

    timestep = int(robot.getBasicTimeStep())

    camera = robot.getDevice("camera")
    camera.enable(timestep)

    display_img = robot.getDevice("display_image")

    keyboard = Keyboard()
    keyboard.enable(timestep)

    set_speed(driver, speed)  # Set initial speed
    set_steering_angle(driver, angle)  # Set initial steering angle

    while robot.step() != -1:
        image = get_image(camera)
        if image is not None:
            grey_image = greyscale_cv2(image)
            display_image(display_img, grey_image)

        regulate(driver, camera)  # Integrate lane detection into the control loop

        key = keyboard.getKey()
        if key == Keyboard.UP:
            set_speed(driver, speed + 5)
        elif key == Keyboard.DOWN:
            set_speed(driver, speed - 5)
        elif key == Keyboard.RIGHT:
            set_steering_angle(driver, angle + 0.1)
        elif key == Keyboard.LEFT:
            set_steering_angle(driver, angle - 0.1)

    if SHOW_IMAGE_WINDOW:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
