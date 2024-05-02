import cv2

img_path = '../../img'

img = cv2.imread(f'{img_path}/lena.jpg', -1)
print(img)

cv2.imshow('lena', img)
k = cv2.waitKey(0) & 0xFF

if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite(f'{img_path}/lena_copy.png', img)




