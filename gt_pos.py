import numpy as np
# from circle_detection_hough2d import main


def hole_pos(pixel_cor, K, rad_pixel, rad_gt, zc=None, end_mat=None):
    f = (K[0, 0] + K[1, 1])/2
    if zc is None:
        zc = f * rad_gt / rad_pixel
    pixel_cor_ = np.concatenate((pixel_cor.reshape(2, 1), np.array([[1]])), 0)
    cam_cor = np.linalg.inv(K) @ (zc * pixel_cor_)
    cam_cor = np.append(cam_cor, np.array([[1]]), axis=0)

    if end_mat is not None:
        hole_world = end_mat @ cam_cor
        print('Result on world coordinate!')
        return hole_world
    print('Result on camera coordinate!')
    return cam_cor


# if __name__ == "__main__":

    # K = np.asarray([[3498.76987761733, 0, 2732.99164070376], [0, 3499.62038882874, 1854.24867441690], [0, 0, 1]], np.float32)
    # path = "C:/Users/liboyan/Desktop/testing/1.2.bmp"
    #
    # center1, center2 = main(path, 20, 75, 40)
    # middle_pt = (center1 + center2)/2
    # pixel_cor = middle_pt[:2]
    # rad_pixel = middle_pt[2]
    #
    # rad_gt = 0.007/2
    #
    # hole_cam_cor = hole_pos(pixel_cor, K, rad_pixel, rad_gt)
    #
    # print(hole_cam_cor)

