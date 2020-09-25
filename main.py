import imutils
import cv2
image = cv2.imread("C:/Users/hargar/Desktop/Python shiz/git/FaceStabilizer/1.jpg")
print(image.shape)
(h, w, d) = image.shape
print("width={}, height={}, depth={}".format(w, h, d))
cv2.imshow("Image", image)
cv2.waitKey(0)

#roi = image[60:160, 320:420]
#resized = cv2.resize(image, (200, 200))
#resized = imutils.resize(image, width=100)
#rotated = imutils.rotate(image, -45)
#blurred = cv2.GaussianBlur(image, (11, 11), 0)

output = image.copy()
cv2.circle(output, (300, 150), 20, (255, 0, 0), -1)
cv2.imshow("Circle", output)

cv2.waitKey(0)
