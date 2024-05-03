from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2



# Constants for lane detection and control
CONTROL_COEFFICIENT = 0.005
SHOW_IMAGE_WINDOW = True

# Initial angle and speed settings
manual_steering = 0
steering_angle = 0
angle = 0.0
speed = 20  # Set initial cruising speed to 30 km/h

def get_image(camera):
    raw_image = camera.getImage()
    if raw_image is None:
        print("Camera image not available.")
        return None
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    # Convert RGBA to RGB if necessary
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
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
    img = get_image(camera)
    if img is None or img.size == 0:
        print("No image data to process.")
        return

    height, width, _ = img.shape
    roi_height_start = int(height * 0.6)
    img_slice = img[roi_height_start:, :]

    gray_img = cv2.cvtColor(img_slice, cv2.COLOR_RGB2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edges = cv2.Canny(blur_img, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=40, maxLineGap=100)
    lines_image = np.zeros((height - roi_height_start, width, 3), dtype=np.uint8)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(lines_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow('Detected Lines', lines_image)
    else:
        print("No lines detected.")

    cv2.imshow('Edges', edges)
    cv2.imshow('ROI', img_slice)
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

        current_speed = driver.getCurrentSpeed()  # Retrieve the current speed of the vehicle
        regulate(driver, camera)  # Integrate lane detection into the control loop with current speed

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