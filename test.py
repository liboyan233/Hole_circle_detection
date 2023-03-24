# import cv2
# import numpy as np
#
# img = cv2.imread('hole.jpg',0)
# img = cv2.medianBlur(img,5)
# cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
#
# circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1, 50,
#                              param1=80,param2=20,minRadius=3,maxRadius=25)
#
#
# circles = np.uint16(np.around(circles))
# for i in circles[0,:]:
#     # Dibuja la circusnferencia del círculo
#     cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
#     # dibuja el centro del círculo
#     cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
#
# cv2.imshow('círculos detectados',cimg)
# cv2.waitKey(0)
# # cv2.destroyAllWindows()

import numpy as np
import cv2


img = cv2.imread("C:/Users/liboyan/Desktop/Industrial PCB Image Dataset/1.bmp",1)

circle_gt = np.array([
[252.461,	228.337, 73.3911,	72.1146,	-1.50459],
[252.868,	228.896,	129.846,	128.368,	-0.42123],
[253.032,	228.051,	145.134,	143.531,	1.1756],
[254.885,	229.096,	200.762,	198.474,	-0.023687],
])

for i in range(4):
    radius = (circle_gt[i, 2] + circle_gt[i, 3])/2
    cv2.circle(img, (int(circle_gt[i, 0]), int(circle_gt[i, 1])), int(radius), (255, 0, 0), 2)


cv2.imshow("circle",img)

cv2.waitKey(0)