import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


path ='um.jpg'

img=cv.imread(path)

hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)




# got upper and lower range of colors in hsv (jupyter notebook test.ipynb)
blue_mask = cv.inRange(hsv,(100,150,0),(140,255,255))   

green_mask = cv.inRange(hsv,(40,70,70),(80,255,255))

orange_mask = cv.inRange(hsv,(13,100,20),(24,255,255))

pink_mask = cv.inRange(hsv,(145,100,20),(165,255,255))

light_pink_mask = cv.inRange(hsv,(131,20,20),(169,109,240))

red_mask1 = cv.inRange(hsv,(0,100,20),(10,255,255))
red_mask2 = cv.inRange(hsv,(170,100,20),(180,255,255))
red_mask = red_mask1 + red_mask2

blue = cv.bitwise_and(img,img,mask=blue_mask)
green = cv.bitwise_and(img,img,mask=green_mask)
orange = cv.bitwise_and(img,img,mask=orange_mask)
pink = cv.bitwise_and(img,img,mask=pink_mask)
light_pink = cv.bitwise_and(img,img,mask=light_pink_mask)
red = cv.bitwise_and(img,img,mask=red_mask)



cv.imshow('original image',img)
cv.imshow('blue mask',blue)
cv.imshow('green mask',green)
cv.imshow('orange mask',orange)
cv.imshow('pink mask',pink)
cv.imshow('light pink mask',light_pink)
cv.imshow('red mask',red)

cv.waitKey(0)
cv.destroyAllWindows()



