import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import gt_pos as gp
from scipy.spatial.transform import Rotation as R
import grabimage as grab

"""problem: use two circle for detection, but can't tell between"""
def foam_localization(path=None, end=None):
    C_Min_Rad = 50
    C_Max_Rad = 200
    L_Num = 2
    C_Num = 2
    theta = 30 / 180 * np.pi
    scale = 0.7

    Len_circles_gt = 89 - 18

    # end = [0.178, -0.56, 0.5, 180, 0, 90]

    if path is None:
        path = "C:/Users/liboyan/Desktop/cv_task_add/test4.bmp"

    src = cv2.imread(path)
    img = src.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 进行中值滤波
    dst_img = cv2.medianBlur(img_gray, 7)

    circle = cv2.HoughCircles(dst_img, cv2.HOUGH_GRADIENT, 1, 50,
                             param1=120, param2=40, minRadius=C_Min_Rad, maxRadius=C_Max_Rad)
    print('hole center:', circle)

    for count, i in enumerate(circle[0, :]):  # 遍历矩阵的每一行的数据
        # print(circle)
        # 绘制圆形
        if count < C_Num:
            cv2.circle(img, (int(i[0]), int(i[1])), int(i[2]), (255, 0, 0), 3)
            # 绘制圆心
            cv2.circle(img, (int(i[0]), int(i[1])), 10, (255, 0, 0), -1)

    """try with houghlines"""
    # edges detection with Canny method
    # edges = cv2.Canny(dst_img, threshold1=50, threshold2=200)
    # HoughLines()函数
    # lines_p = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)
    #
    # for i in range(len(lines_p)):
    #     if i < L_Num:
    #         x_1, y_1, x_2, y_2 = lines_p[i][0]
    #         cv2.line(img, (x_1, y_1), (x_2, y_2), (0, 255, 0), 2)
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    circle_len = np.linalg.norm(circle[0, 0, :2] - circle[0, 1, :2])
    if circle[0, 0, 1] > circle[0, 1, 1]:
        vec = (circle[0, 1, :2] - circle[0, 0, :2]) / circle_len
        aim_loc = circle[0, 1, :2] + rot @ vec * circle_len * scale
    else:
        vec = (circle[0, 0, :2] - circle[0, 1, :2]) / circle_len
        aim_loc = circle[0, 0, :2] + rot @ vec * circle_len * scale

    print('aim_loc:', aim_loc)
    cv2.circle(img, (int(aim_loc[0]), int(aim_loc[1])), 10, (0, 0, 255), -1)

    cv2.namedWindow("final_result", cv2.WINDOW_NORMAL)
    cv2.imshow("final_result",img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if end is not None:
        K = np.asarray([[3498.76987761733, 0, 2732.99164070376], [0, 3499.62038882874, 1854.24867441690], [0, 0, 1]],
                       np.float32)

        len_pixel = np.linalg.norm(circle[0, 0, :2] - circle[0, 1, :2])
        len_gt = Len_circles_gt / 1000

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

        gt_cor = gp.hole_pos(aim_loc, K, len_pixel, len_gt, end_mat=mat_T @ mat_grib2eye)
        print('world coordinate:', '\n', gt_cor[:3])


if __name__ == "__main__":
    try:
        hhv = grab.HHV()
        path = "AfterConvert_RGB0.jpg"
    except:
        print('camera connection error')
        path = "C:/Users/liboyan/Desktop/cv_task_add/test5.bmp"
    end = [-0.3, 0.7, 0.550, 180, 0, -45]
    foam_localization(path=path, end=end)
