import imutils
import cv2
import numpy as np
from transform import four_point_transform
from colorCorrect import simplest_cb, adjust_gamma

debug = 0
output = 1

def splitPhotos(path):
    image = cv2.imread(path)
    #Crop left noise
    image = image[0:, 15:]
    #Add border
    image= cv2.copyMakeBorder(image,50,50,50,50,cv2.BORDER_CONSTANT,value=[255,255,255])
    #cv2.imshow('bordered image',image)
    #cv2.waitKey(0)
    if debug == 1:
        image = imutils.resize(image, height = 1000)
        cv2.imshow('image',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    orig = image.copy()

    #Create a mask
    lower_black = np.array([240,240,240], dtype = "uint16")
    upper_black = np.array([255,255,255], dtype = "uint16")
    black_mask = cv2.inRange(image, lower_black, upper_black)
    if debug == 1:
        cv2.imshow('black_mask',black_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #Erode and Dilate to remove noise
    mask = cv2.dilate(black_mask, None, iterations=6)
    if debug == 1:
        cv2.imshow('mask',mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    mask = cv2.erode(mask, None, iterations=6)
    if debug == 1:
        cv2.imshow('mask',mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #Blur to smoothen lines
    blurred = cv2.GaussianBlur(mask, (3, 3), 0)
    if debug == 1:
        cv2.imshow('final mask',blurred)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #Find contours and keep the largets ones (except the largest one and remove the unwanted ones)
    cnts = cv2.findContours(blurred.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted( (c for c in cnts if cv2.arcLength(c, True) > 500) , key = cv2.contourArea, reverse = True)[1:]

    #Loop over the contours
    print( "Number of Photos:" + str(len(cnts)) )
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        print("Perimeter:" + str(peri) )
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
        # if our approximated contour has four points, then we
        # can assume that we have a photo
        if len(approx) == 4 :
            print("Photo: True")
            screenCnt = approx
            photo = four_point_transform(orig, screenCnt.reshape(4, 2))
            out = simplest_cb(photo, 1)
            if output == 1:
                temp1 = imutils.resize(photo, height = 600)
                cv2.imshow("Cropped", temp1)
                temp2 = imutils.resize(out, height = 600)
                cv2.imshow("Corrected", temp2)
                cv2.waitKey(0)
            #print(cv2.imwrite(".\\resources\\a.jpg", photo))

    if debug == 1:
        cv2.imshow("photo", image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
