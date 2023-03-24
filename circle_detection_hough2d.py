import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
from metric import Circle
import os
import circle_detection_FFT as fft
from gt_pos import hole_pos


def main(file_path, circle_num=1, max_rad=200, min_rad=0, file_name=None):
    print('-----------------------------------------------')
    src = cv2.imread(file_path)
    # print(src.shape)
    img = src.copy()
    print(np.shape(img))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 进行中值滤波
    dst_img = cv2.medianBlur(img_gray, 7)

    up = 1500
    side = 1500

    dst_img = dst_img[up:2500, side:3000]

    # 霍夫圆检测
    circle = cv2.HoughCircles(dst_img, cv2.HOUGH_GRADIENT, 1, 20,
                             param1=60, param2=21, minRadius=min_rad, maxRadius=max_rad)  # 60, 30

    circle_det = []

    if circle is None:
        max_circle = 0
    else:
        max_circle = np.shape(circle)[1]

    if max_circle < circle_num:
        circle_num = max_circle
    for i in range(circle_num):
        circle_det.append(Circle(circle[0, i, 0], circle[0, i, 1], circle[0, i, 2]))
        # print(circle[0, i, 0], circle[0, i, 1], circle[0, i, 2])

    # 显示图像
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 8), dpi=100)
    # axes[0].imshow(src[:, :, ::-1])
    # axes[0].set_title("Original Figure")
    # axes[1].imshow(img[:, :, ::-1])
    # axes[1].set_title("Figure after Hough Detection")
    # plt.show()

    center1 = circle[0, 0]

    """ find nearby center, method 1 """
    # count = 1
    # center2 = None
    # while count < len(circle[0]):
    #     if (center1[0] - circle[0, count, 0])**2 + (center1[1] - circle[0, count, 1])**2 <= center1[2]**2*2 and abs(center1[2] - circle[0, count, 2]) <= 10:
    #         center2 = circle[0, count]
    #         break
    #     count += 1

    """ using black pixel inside the first circle """
    # rad = int(circle[0, 0, 2])+1
    # new_img = np.zeros((2*rad+1, 2*rad+1))
    # print()
    # for i in range(-rad, rad+1):
    #     for j in range(-rad, rad+1):
    #         if i**2 + j**2 <= rad**2 and dst_img[i+int(circle[0, 0, 0]), j+int(circle[0, 0, 1])] > 0.8:
    #             new_img[i+rad, j+rad] = 1
    # cv2.namedWindow("circle", cv2.WINDOW_NORMAL)
    # cv2.imshow("circle", new_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    """ method 2 """
    rad = int(circle[0, 0, 2]) + 1
    dst_img_new = dst_img[int(circle[0, 0, 1]) - rad:int(circle[0, 0, 1]) + rad, int(circle[0, 0, 0]) - rad:int(circle[0, 0, 0]) + rad]
    overlapping = np.count_nonzero(dst_img_new < 30) / (np.pi*rad**2)
    # print('covering_rate:', overlapping)

    overlap_thre_lo = 0.4
    overlap_thre_up = 0.7
    rad_tole = 5
    if overlapping < overlap_thre_lo:  # maybe the second detector can work, but not robust
        print('Warning! Holes overlapping is unqualified')
    elif overlapping > overlap_thre_up:
        print('Two Holes overlap well!')

    # _, binary = cv2.threshold(dst_img_new, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 霍夫圆检测
    circle2 = cv2.HoughCircles(dst_img_new, cv2.HOUGH_GRADIENT, 1, 20,
                             param1=20, param2=8, minRadius=int(circle[0, 0, 2] - rad_tole), maxRadius=int(circle[0, 0, 2] + rad_tole))  # 60, 30

    if np.shape(circle2[0])[0] < 2:
        center2 = None
        if overlap_thre_up > overlapping > overlap_thre_lo:
            print('Error! Second hole not found.')
    else:
        center2 = circle2[0, 1]

    center1[0] += + side
    center1[1] += + up
    print('first detection', circle[0, 0, :])

    if center2 is not None:
        center2[0] += int(center1[0]) - rad
        center2[1] += int(center1[1]) - rad
        print('second detection', [center2[0], center2[1], circle[0, 0, 2]])

    # """using FFT method""" -> big fault!
    # center_fft, rad_fft = fft.main(dst_img_new, 2, max_rad=circle2[0, 0, 2]+1, min_rad=circle2[0, 0, 2], ada_version=True)
    # print('second detection:', center_fft, rad_fft)
    # center2 = [center_fft[0][0], center_fft[0][1], rad_fft[0]]
    # if center2 is not None:
    #     center2[0] += int(circle[0, 0, 0]) - rad
    #     center2[1] += int(circle[0, 0, 1]) - rad
    # print(center2)

    """show result figure"""
    # 将检测结果绘制在图像上
    # count = 0
    # for i in circle[0, :]:  # 遍历矩阵的每一行的数据
    #     count += 1
    #     if count <= circle_num:
    #         # 绘制圆形
    #         cv2.circle(img, (int(i[0]+side), int(i[1])+up), int(i[2]), (0, 255, 0), 6)
    #         # 绘制圆心
    #         # cv2.circle(dst_img, (int(i[0]), int(i[1])), 10, (255, 0, 0), -1)

    cv2.circle(img, (int(center1[0]), int(center1[1])), int(center1[2]), (255, 0, 0), 10)
    # 绘制圆心
    # cv2.circle(img, (int(center1[0]), int(center1[1])), 10, (255, 0, 0), -1)

    if center2 is not None:
        cv2.circle(img, (int(center2[0]), int(center2[1])), int(center2[2]), (255, 255, 0), 10)
        # 绘制圆心
        # cv2.circle(dst_img, (int(center2[0]), int(center2[1])), 10, (255, 0, 0), -1)

    if file_name is None:
        cv2.namedWindow("circle", cv2.WINDOW_NORMAL)
        cv2.imshow("circle",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.imwrite(file_name + '_result' + '.png', img)

    return center1, center2

if __name__ == "__main__":
    # path = "C:/Users/liboyan/Desktop/testing/1.2.bmp"
    # # path_org = "C:/Users/liboyan/Desktop/"
    # # for i in range(4):
    #     # path = path_org + str(3.5) + '.png'
    # main(path, 20, 75, 40)

    path = "C:/Users/liboyan/Desktop/testing"
    files = os.listdir(path)  # 得到文件夹下的所有文件名称

    # for i, file in enumerate(files):
    #
    #     NewName = os.path.join(path, file[:3] + '.bmp')
    #     OldName = os.path.join(path, file)
    #     os.rename(OldName, NewName)

    for file in files:  # 遍历文件夹

        img_path = path + '/' + file
        print(img_path)
        center1, center2 = main(img_path, 10, 80, 40, file)

        K = np.asarray([[3498.76987761733, 0, 2732.99164070376], [0, 3499.62038882874, 1854.24867441690], [0, 0, 1]],
                       np.float32)
        if center2 is None:
            middle_pt = center1
        else:
            middle_pt = (center1 + center2) / 2
        pixel_cor = middle_pt[:2]
        rad_pixel = middle_pt[2]

        rad_gt = 0.007 / 2

        hole_cam_cor = hole_pos(pixel_cor, K, rad_pixel, rad_gt)
        print('Hole world position:', '\n', hole_cam_cor)


