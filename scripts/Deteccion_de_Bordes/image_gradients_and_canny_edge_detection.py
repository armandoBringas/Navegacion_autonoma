import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../../img/sudoku.jpeg', cv2.IMREAD_GRAYSCALE)

lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
lap = np.uint8(np.absolute(lap))

sobel_X = cv2.Sobel(img, cv2.CV_64F, 1, 0)
sobel_Y = cv2.Sobel(img, cv2.CV_64F, 0, 1)
sobel_X = np.uint8(np.absolute(sobel_X))
sobel_Y = np.uint8(np.absolute(sobel_Y))

sobel_combined = cv2.bitwise_or(sobel_X, sobel_Y)

titles = ['image', 'Laplacian', 'Sobel X', 'Sobel Y', 'Sobel combined']
images = [img, lap, sobel_X, sobel_Y, sobel_combined]

n_images = len(images)

for i in range(n_images):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()

