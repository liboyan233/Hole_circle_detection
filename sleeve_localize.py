import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import gt_pos as gp
from scipy.spatial.transform import Rotation as R
import grabimage as grab

"""This program work for camera at 800mm height"""
C1_Min_Rad = 70
C1_Max_Rad = 85

# robot hand end pose
end = [-0.44, 0.243, 0.6, 180, 0, 111]  # for base, use meter and degree as unit
# end = [-0.32, 1, 0.5, 180, 0, -55]
#
path = "C:/Users/liboyan/MVS/Data/Image_20230609223720249.bmp"
# try:
#     hhv = grab.HHV()
#     path = "AfterConvert_RGB0.jpg"
# except:
#     print('camera connection error')
#     path = "AfterConvert_RGB0.jpg"

src = cv2.imread(path)
h, w = src.shape[:2]
img = src.copy()
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 进行中值滤波
dst_img = cv2.medianBlur(img_gray, 7)

# 霍夫圆检测
circle = cv2.HoughCircles(dst_img, cv2.HOUGH_GRADIENT, 1, 50,
                         param1=70, param2=30, minRadius=C1_Min_Rad, maxRadius=C1_Max_Rad)
print('hole center:', circle)
# 将检测结果绘制在图像上
for i in circle[0, :]:  # 遍历矩阵的每一行的数据
    # print(circle)
    # 绘制圆形
    cv2.circle(img, (int(i[0]), int(i[1])), int(i[2]), (255, 0, 0), 3)
    # 绘制圆心
    cv2.circle(img, (int(i[0]), int(i[1])), 10, (255, 0, 0), -1)

middle_idx = -1
min_dis_to_mid = np.inf
for i in range(len(circle[0, :])):
    dis = ((circle[0, i, 0] - w / 2) ** 2 + (circle[0, i, 1] - h / 2) ** 2) ** 0.5
    if dis < min_dis_to_mid:
        min_dis_to_mid = dis
        middle_idx = i

center = circle[0, middle_idx, :2]
r = circle[0, middle_idx, 2]

cv2.circle(img, (int(center[0]), int(center[1])), int(r), (255, 0, 0), 3)
# 绘制圆心
cv2.circle(img, (int(center[0]), int(center[1])), 10, (0, 255, 0), -1)


cv2.namedWindow("final_result", cv2.WINDOW_NORMAL)
cv2.imshow("final_result",img)

cv2.waitKey(0)
cv2.destroyAllWindows()

K = np.asarray([[3498.76987761733, 0, 2732.99164070376], [0, 3499.62038882874, 1854.24867441690], [0, 0, 1]],
                       np.float32)

rot_end = R.from_euler('Z', end[5], degrees=True).as_matrix() @ R.from_euler('Y', end[4], degrees=True)\
    .as_matrix() @ R.from_euler('X', end[3], degrees=True).as_matrix()

mat_T = np.zeros([4, 4])
mat_T[:3, :3] = rot_end
mat_T[0, 3] = end[0]
mat_T[1, 3] = end[1]
mat_T[2, 3] = end[2]
mat_T[3, 3] = 1

mat_grib2eye = np.array([[0.69433217, -0.71964302, -0.00409354, 0.15782011],
                    [0.71960065,  0.69419913,  0.01620134,  0.16356692],
                    [-0.00881745, -0.01419483,  0.99986037,  0.15466117],
                    [0, 0, 0, 1]])

len_gt = 0.063
len_pixel = ((1926.5 - 2271.5)**2 + (970.5 - 1345.5)**2)**0.5
f = (K[0, 0] + K[1, 1]) / 2
print(f * len_gt / len_pixel)
zc = f * len_gt / len_pixel + (end[2] - 0.8)

gt_cor = gp.hole_pos(center, K, len_pixel, len_gt, end_mat=mat_T @ mat_grib2eye, zc=zc)  # only work for 800mm height
print('world coordinate:', '\n', gt_cor[:3])

