import imutils
import cv2
import numpy as np
from transform import four_point_transform

# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
#image = cv2.imread(args["image"])
image = cv2.imread("C:/Users/hargar/Desktop/Python shiz/git/FaceStabilizer/resources/11.jpg")
ratio = image.shape[0] / 500.0
orig = image.copy()

image = imutils.resize(image, height = 800)
cv2.imshow('image',image)
cv2.waitKey(0)

lower_black = np.array([240,240,240], dtype = "uint16")
upper_black = np.array([255,255,255], dtype = "uint16")
black_mask = cv2.inRange(image, lower_black, upper_black)
cv2.imshow('black_mask',black_mask)
cv2.waitKey(0)

mask = cv2.dilate(black_mask, None, iterations=1)
cv2.imshow('mask',mask)
cv2.waitKey(0)
mask = cv2.erode(mask, None, iterations=1)
cv2.imshow('mask',mask)
cv2.waitKey(0)

blurred = cv2.GaussianBlur(mask, (1, 1), 0)
cv2.imshow('gray',blurred)
cv2.waitKey(0)

# show the original image and the edge detected image
print("STEP 1: Edge Detection")
cv2.destroyAllWindows()

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cnts = cv2.findContours(blurred.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[1:5]

print("STEP 2: Find contours of paper")

# loop over the contours
print( len(cnts) )
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    print("Size:" + str(peri) )
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # if our approximated contour has four points, then we
    # can assume that we have found our screen

    if len(approx) == 4:
        screenCnt = approx
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
        warped = four_point_transform(image, screenCnt.reshape(4, 2))
        cv2.imshow("Outline", warped)
        cv2.waitKey(0)

cv2.destroyAllWindows()
