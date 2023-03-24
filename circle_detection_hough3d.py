from __future__ import division
import cv2
import numpy as np
import time


def fill_acc_array(x0,y0,radius,acc_array):
    x = radius
    y=0
    decision = 1-x

    acc_shape = np.shape(acc_array)
    # print(acc_shape)
    height = acc_shape[0]
    width = acc_shape[1]
    add = 1/radius
    while(y<x):
        if(x + x0<height and y + y0<width):
            acc_array[ x + x0,y + y0,radius]+=add # Octant 1
        if(y + x0<height and x + y0<width):
            acc_array[ y + x0,x + y0,radius]+=add # Octant 2
        if(-x + x0<height and y + y0<width):
            acc_array[-x + x0,y + y0,radius]+=add # Octant 4
        if(-y + x0<height and x + y0<width):
            acc_array[-y + x0,x + y0,radius]+=add # Octant 3
        if(-x + x0<height and -y + y0<width):
            acc_array[-x + x0,-y + y0,radius]+=add # Octant 5
        if(-y + x0<height and -x + y0<width):
            acc_array[-y + x0,-x + y0,radius]+=add # Octant 6
        if(x + x0<height and -y + y0<width):
            acc_array[ x + x0,-y + y0,radius]+=add # Octant 8
        if(y + x0<height and -x + y0<width):
            acc_array[ y + x0,-x + y0,radius]+=add # Octant 7
        y+=1
        if(decision<=0):
            decision += 2 * y + 1
        else:
            x=x-1
            decision += 2 * (y - x) + 1
    return acc_array


def main(file_path, circle_num=4, max_rad=110, min_rad=35):
    original_image = cv2.imread(file_path,1)

    # down_num = 1
    # for i in range(down_num):
    #     original_image = cv2.pyrDown(original_image)

    #gray_image = cv2.imread('Sample_Input.jpg',0)
    cv2.imshow('Original Image',original_image)

    output = original_image.copy()
    # img_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    #Gaussian Blurring of Gray Image
    blur_image = cv2.GaussianBlur(original_image,(3,3),0)
    # cv2.imshow('Gaussian Blurred Image',blur_image)

    #Using OpenCV Canny Edge detector to detect edges
    edged_image = cv2.Canny(blur_image,75,150)
    # cv2.imshow('Edged Image', edged_image)

    height,width = edged_image.shape
    radii = max_rad

   #  acc_array = np.zeros((height,width,radii))  # height*width*radii

    filter3D = np.zeros((30,30,radii))  # 30*30*radii
    filter3D[:,:,:]=1

    start_time = time.time()

    edges = np.where(edged_image==255)


    for i in range(len(edges[0])):
        x=edges[0][i]
        y=edges[1][i]
        for radius in range(min_rad, max_rad):
            acc_array = fill_acc_array(x, y, radius, acc_array)

    i=0
    j=0

    circle_det = []

    for i in range(circle_num):
        index = np.argwhere(acc_array == np.amax(acc_array))
        acc_array[index[0, 0]-10:index[0, 0]+10, index[0, 1]-10:index[0, 1]+10, index[0, 2]-5:index[0, 2]+5] = 0
        cv2.circle(output,(index[0, 1],index[0, 0]),index[0, 2],(0,255,0),2)
        re = np.array([index[0, 1], index[0, 0], index[0, 2]])
        print(re)
        acc_array[index[0, :]] = 0
        # circle_det.append()

    # while(i<height-30):
    #     while(j<width-30):
    #         filter3D=acc_array[i:i+30,j:j+30,:]*filter3D
    #         max_pt = np.where(filter3D==filter3D.max())
    #         a = max_pt[0]
    #         b = max_pt[1]
    #         c = max_pt[2]
    #         b=b+j
    #         a=a+i
    #         if(filter3D.max()>40):
    #             for k in range(len(a)):
    #                 print((b[k],a[k]),c[k])
    #                 cv2.circle(output,(b[k],a[k]),c[k],(0,255,0),2)
    #         j=j+30
    #         filter3D[:,:,:]=1
    #     j=0
    #     i=i+30

    end_time = time.time()
    time_taken = end_time - start_time

    print ('Time taken for execution',time_taken)
    # cv2.imshow('Detected circle',output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return circle_det


if __name__ == "__main__":
    path = "C:/Users/liboyan/Desktop/Industrial_PCB_Image_Dataset/Industrial PCB Image Dataset/1.bmp"
    main(path)