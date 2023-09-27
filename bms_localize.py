import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import gt_pos as gp
from scipy.spatial.transform import Rotation as R
import grabimage as grab


C1_Min_Rad = 50
C1_Max_Rad = 65
C2_Min_Rad = 90
C2_Max_Rad = 180

# for target point
theta = 77 / 180 * np.pi
scale = 0.48

mode = 1

# robot hand end pose
end = [0.02, 1, 0.5, 180, 0, -76]  # for base
# end = [-0.32, 1, 0.5, 180, 0, -55]

# targeted BMS Pose -> far away circle -- close circle
target_pos = [0.99999549, -0.00269208]

start = time.time()
# path = "C:/Users/liboyan/MVS/Data/Image_20230301160454849.bmp"
try:
    hhv = grab.HHV()
    path = "AfterConvert_RGB0.jpg"
except:
    print('camera connection error')
    path = "AfterConvert_RGB0.jpg"

src = cv2.imread(path)
img = src.copy()
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 进行中值滤波
dst_img = cv2.medianBlur(img_gray, 7)

# 霍夫圆检测
circle = cv2.HoughCircles(dst_img, cv2.HOUGH_GRADIENT, 1, 50,
                         param1=70, param2=20, minRadius=C1_Min_Rad, maxRadius=C1_Max_Rad)
print('hole center:', circle)
# 将检测结果绘制在图像上
for i in circle[0, :]:  # 遍历矩阵的每一行的数据
    # print(circle)
    # 绘制圆形
    cv2.circle(img, (int(i[0]), int(i[1])), int(i[2]), (255, 0, 0), 3)
    # 绘制圆心
    cv2.circle(img, (int(i[0]), int(i[1])), 10, (255, 0, 0), -1)


# """find rectangle"""
# ret,binary = cv2.threshold(img_gray,120,255,cv2.THRESH_BINARY)
#
# contours,hierarchy = cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# print("Number of contours:" + str(len(contours)))
#
# inten = []
# inten_index = []
# for i in range(len(contours)):
#     x, y, w, h = cv2.boundingRect(contours[i])
#     if w > 200 and h > 200:
#         part = binary[y:y+h, x:x+w]
#
#         intensity = np.sum(part == 0)/(w*h)
#         inten.append(intensity)
#         inten_index.append(i)
#         # print(x, y, w, h,intensity)
#
# """get the best rectangle"""
# contour_index = inten_index[np.argmax(inten)]
# x,y,w,h = cv2.boundingRect(contours[contour_index])
# cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
#
#
# """calculate center"""
# # circle_center = np.sum(circle[0, :, :2], axis=0)/2
# rectangle_center = np.array([x+w/2, y+h/2])
# for i in range(2):
#     width1 = abs(circle[0, 0, 1] - (y + h/2))
#     width2 = abs(circle[0, 1, 1] - (y + h / 2))
#     if width1 < width2:
#         circle_right = circle[0, 0, :2]
#     else:
#         circle_right = circle[0, 1, :2]
# center = rectangle_center/2 + circle_right/2

"""find one bigger center"""
circle2 = cv2.HoughCircles(dst_img, cv2.HOUGH_GRADIENT, 1, 50,
                         param1=140, param2=30, minRadius=C2_Min_Rad, maxRadius=C2_Max_Rad)
i = circle2[0, 0]
print(i)
# 绘制圆形
cv2.circle(img, (int(i[0]), int(i[1])), int(i[2]), (255, 255, 0), 3)
# 绘制圆心
cv2.circle(img, (int(i[0]), int(i[1])), 10, (255, 0, 0), -1)

"""choose the farther smaller center and calculate pos we need"""

rot = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])

dis1 = np.linalg.norm(circle[0, 0, :2] - circle2[0, 0, :2])
dis2 = np.linalg.norm(circle[0, 1, :2] - circle2[0, 0, :2])

circle_len = np.linalg.norm(circle[0, 0, :2] - circle[0, 1, :2])

if mode == 1:
    if dis1 > dis2:
        vec = (circle[0, 0, :2] - circle[0, 1, :2]) / circle_len
        center = circle[0, 1, :2] + rot @ vec * circle_len * scale
    else:
        vec = (circle[0, 1, :2] - circle[0, 0, :2]) / circle_len
        center = circle[0, 0, :2] + rot @ vec * circle_len * scale
else:
    if dis1 > dis2:
        vec = (circle[0, 1, :2] - circle[0, 0, :2]) / circle_len
        center = circle[0, 0, :2] + rot @ vec * circle_len * scale
    else:
        vec = (circle[0, 0, :2] - circle[0, 1, :2]) / circle_len
        center = circle[0, 1, :2] + rot @ vec * circle_len * scale
# center = circle2[0, 0, :2] + rot @ vec * circle_len * scale
#
end_time = time.time()

print('time used = ', end_time - start)
cv2.circle(img, (int(center[0]), int(center[1])), 10, (0, 0, 255), 8)

cv2.namedWindow("final_result", cv2.WINDOW_NORMAL)
cv2.imshow("final_result",img)

cv2.waitKey(0)
cv2.destroyAllWindows()

K = np.asarray([[3498.76987761733, 0, 2732.99164070376], [0, 3499.62038882874, 1854.24867441690], [0, 0, 1]],
                       np.float32)

len_pixel = np.linalg.norm(circle[0, 0, :2] - circle[0, 1, :2])
len_gt = 72/1000

rot_end = R.from_euler('Z', end[5], degrees=True).as_matrix() @ R.from_euler('Y', end[4],
                                                                             degrees=True).as_matrix() @ R.from_euler(
    'X', end[3], degrees=True).as_matrix()

mat_T = np.zeros([4, 4])
mat_T[:3, :3] = rot_end
mat_T[0, 3] = end[0]
mat_T[1, 3] = end[1]
mat_T[2, 3] = end[2]
mat_T[3, 3] = 1

mat_grib2eye = np.array([[-0.69782755, 0.71577142, - 0.02660799, - 0.1097311],
                         [-0.71560263, - 0.69829843, - 0.01709334, - 0.11668819],
                         [-0.03081524, 0.00711254, 0.99949979, 0.14796457],
                         [0, 0, 0, 1]])

gt_cor = gp.hole_pos(center, K, len_pixel, len_gt, end_mat=mat_T @ mat_grib2eye)
print('world coordinate:', '\n', gt_cor[:3])

target_pos = target_pos / np.linalg.norm(target_pos)

# vec_ = vec.reshape(-1, 1)
# vec_ = np.insert(vec_, 2, 0, axis=0)
# curr_pos = ((mat_T @ mat_grib2eye)[:3, :3] @ vec_)[:2, ]

curr_pos = (mat_T @ mat_grib2eye)[:2, :2] @ vec

print('current_pos:', curr_pos)

adapt_ang = np.arccos(np.sum(target_pos*curr_pos))

if np.cross(curr_pos, target_pos) < 0:
    adapt_ang = -adapt_ang

print('adapt angle is:', end[5] + adapt_ang/np.pi*180)



