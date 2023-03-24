import cv2
import numpy as np


image = cv2.imread("C:/Users/liboyan/Desktop/hole2.jpg")

gray =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret,binary = cv2.threshold(gray,120,255,cv2.THRESH_BINARY)
print(np.shape(binary))

contours,hierarchy = cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours:" + str(len(contours)))

inten = []
inten_index = []
for i in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[i])
    if w > 100 and h > 100:
        part = binary[y:y+h, x:x+w]

        intensity = np.sum(part == 0)/(w*h)
        inten.append(intensity)
        inten_index.append(i)
        print(x, y, w, h,intensity)

"""test"""
# x, y, w, h = cv2.boundingRect(contours[inten_index[4]])
# # binary[y:y+h, x:x+w] = 122
# cv2.rectangle(binary,(x,y),(x+w,y+h),0,3)
# # image[y:y+h, x:x+w,:] = [0,255,0]
# cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),3)

"""binary image"""
cv2.namedWindow("bi", cv2.WINDOW_NORMAL)
cv2.imshow('bi',binary)

"""get the best rectangle"""
contour_index = inten_index[np.argmax(inten)]
x,y,w,h = cv2.boundingRect(contours[contour_index])
cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),3)

"""draw all rectangle"""
# for i in inten_index:
#
#     x,y,w,h = cv2.boundingRect(contours[i])
#     if w>200 and h>200:
#         print(x,y,w,h)
#         cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
#         flag = 1
#     i += 1


cv2.namedWindow("result", cv2.WINDOW_NORMAL)
cv2.imshow("result", image)

cv2.waitKey(0)
cv2.destroyAllWindows()