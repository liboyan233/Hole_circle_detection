import metric as mt
import os
import circle_detection_FFT as fft
import circle_detection_hough2d as hough2d
import circle_detection_hough3d as hough3d
import numpy as np
from metric import metric_VRIc

gt_path = 'C:/Users/liboyan/Desktop/Industrial_PCB_Image_Dataset/Ground_truth'
img_path = 'C:/Users/liboyan/Desktop/Industrial_PCB_Image_Dataset/Industrial PCB Image Dataset'
files = os.listdir(gt_path) #得到文件夹下的所有文件名称
txts = []

detect_method_list = ['hough2d', 'hough3d', 'fft']
detect_method = detect_method_list[1]

result = []

for file in files: #遍历文件夹
    print(file)
    gt_position = gt_path + '/' + file
    file_name = file.split('.')[0]
    img_position = img_path + '/' + file_name + '.bmp'
    circle_num = 0
    max_rad = 0
    min_rad = np.inf
    circle_gt_list = []

    with open(gt_position, "r",encoding='utf-8') as f:    #打开文件
        data = f.readlines()  # txt中所有字符串读入data
        circle = False
        for line in data:
            odom = line.split()  # 将单个数据分隔开存好
            if circle is False:
                circle_num = int(odom[0])
                circle = True
            else:
                odom = list(map(float, odom))
                circle_gt_list.append(mt.Circle(odom[0], odom[1], (odom[2]+odom[3])/2))
                if (odom[2]+odom[3])/2 > max_rad:
                    max_rad = (odom[2]+odom[3])/2
                if (odom[2]+odom[3])/2 < min_rad:
                    min_rad = (odom[2]+odom[3])/2

    max_rad = int(max_rad) + 5
    min_rad = int(min_rad) - 5

    if detect_method == 'hough2d':
        circle_det_list = hough2d.main(img_position, circle_num, max_rad, min_rad)  # image, circle_num, max_radius, min_radius
    elif detect_method == 'hough3d':
        circle_det_list = hough3d.main(img_position, circle_num, max_rad, min_rad)
    elif detect_method == 'fft':
        circle_det_list = fft.main(img_position, circle_num, max_rad, min_rad)

    VRIc = metric_VRIc(circle_det_list, circle_gt_list)
    result.append(VRIc)
    print(detect_method + ': ' + str(VRIc))
    print('\n' + '=============================================')

print('Mean_result:', sum(result)/len(result))

