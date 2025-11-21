import cv2

img = cv2.imread('images.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5), 0.7)

canny = cv2.Canny(blur, 50, 150)

contours, h = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

img_contours = img.copy()

cv2.drawContours(img_contours, contours,-1, (0, 255, 0), 8)


cv2.imshow('Contours', img_contours)

cv2.waitKey(0)
cv2.destroyAllWindows()
