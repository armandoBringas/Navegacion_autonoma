from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2

# Constants for lane detection and control
CONTROL_COEFFICIENT = 0.010
SHOW_IMAGE_WINDOW = True

# Initial angle and speed settings
manual_steering = 0
steering_angle = 0
angle = 0.0
speed = 45  # Set initial cruising speed to 45 km/h


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
    image_rgb = np.dstack((image, image, image))
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
    angle = wheel_angle  # Update the global angle
    driver.setSteeringAngle(angle)


def process_image(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edges = cv2.Canny(blur_img, 40, 60)
    return edges

def detect_lines(edges):
    return cv2.HoughLinesP(edges, 2, np.pi/180, 40, minLineLength=50, maxLineGap=10)


def draw_lines(img_slice, lines):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_slice, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw lines in green


def calculate_steering(lines, width):
    if lines is not None:
        best_line = min(lines, key=lambda line: (
            abs((line[0][0] + line[0][2]) / 2 - width / 2) +
            10 * abs(line[0][2] - line[0][0]) / (abs(line[0][3] - line[0][1]) + 1)
        ))
        return calculate_steering_correction(best_line, width)
    return 0


def calculate_steering_correction(best_line, width):
    x1, y1, x2, y2 = best_line[0]
    if (x2 - x1) != 0:
        slope = (y2 - y1) / (x2 - x1)
        angle_from_vertical = np.arctan(slope) * 180 / np.pi
        center_distance = abs((x1 + x2) / 2 - width / 2)
        adjustment_factor = max(1, center_distance / (width / 4))
        return -angle_from_vertical * CONTROL_COEFFICIENT * adjustment_factor
    return 0


def regulate(driver, camera):
    img = get_image(camera)
    if img is None or img.size == 0:
        print("No image data to process.")
        return

    height, width, _ = img.shape
    img_slice = img[int(height * 0.6):, :]
    edges = process_image(img_slice)
    lines = detect_lines(edges)
    draw_lines(img_slice, lines)
    steering_correction = calculate_steering(lines, width)
    driver.setSteeringAngle(steering_correction)
    print(f"Steering adjusted by: {steering_correction} degrees")
    cv2.imshow('Edges', edges)
    cv2.imshow('ROI with Lines', img_slice)
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

    set_speed(driver, speed)
    set_steering_angle(driver, angle)

    while robot.step() != -1:
        image = get_image(camera)
        if image is not None:
            grey_image = greyscale_cv2(image)
            display_image(display_img, grey_image)

        current_speed = driver.getCurrentSpeed()
        regulate(driver, camera)

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
