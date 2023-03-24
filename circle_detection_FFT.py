'''
Created on May 30, 2012
@author: vinnie
'''

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches
from scipy.signal import fftconvolve
from scipy.ndimage import filters
from metric import Circle

_ANNULUS_WIDTH = 5
_EDGE_THRESHOLD = 0.005
_NEG_INTERIOR_WEIGHT = 1.1


def _detectEdges(image, threshold):
    """
    Sobel edge detection on the image
    """
    # sobel filter in x and y direction
    image = filters.sobel(image, 0) ** 2 + filters.sobel(image, 1) ** 2
    image -= image.min()

    # make binary image
    image = image > image.max() * threshold
    image.dtype = np.int8

    return image


def _makeAnnulusKernel(outer_radius, annulus_width):
    """
    Create an annulus with the given inner and outer radii

    Ex. inner_radius = 4, outer_radius = 6 will give you:


    """
    grids = np.mgrid[-outer_radius:outer_radius + 1, -outer_radius:outer_radius + 1]

    # [j][i] = r^2
    kernel_template = grids[0] ** 2 + grids[1] ** 2

    # get boolean value for inclusion in the circle
    outer_circle = kernel_template <= outer_radius ** 2
    inner_circle = kernel_template < (outer_radius - annulus_width) ** 2

    # back to integers
    outer_circle.dtype = inner_circle.dtype = np.int8
    inner_circle = inner_circle * _NEG_INTERIOR_WEIGHT
    annulus = outer_circle - inner_circle
    return annulus


def _detectCircles(image, radii, annulus_width):
    """
    Perfrom a FFT Convolution over all the radii with the given annulus width.
    Smaller annulus width = more precise
    """
    acc = np.zeros((radii.size, image.shape[0], image.shape[1]))

    for i, r in enumerate(radii):
        C = _makeAnnulusKernel(r, annulus_width)
        acc[i, :, :] = fftconvolve(image, C, 'same')

    return acc


def _iterativeDetectCircles(edges, image):
    """
    TODO: finish this

    The idea:
    Start with an annulus with a large radius.
    Split this into 2 annuli of equal area. The annulus width will be different
    Find the annulus with a higher signal in the image
    Repeat the process for the higher signal,
    until a minumum annulus width is reached (preferrably 1)

    Initialize:
    large radius = min(image.shape)/2 ... actually maybe a little less
    min radius is some small number, not too small to avoid high match to noise

    TODO: the signals for each annulus have to be comparable
             solution? -> normalize to the area of the annulus
    """

    return


def _displayResults(image, edges, center, radius, output=None):
    """
    Display the accumulator for the radius with the highest votes.
    Draw the radius on the image and display the result.
    """

    # display accumulator image
    plt.gray()
    fig = plt.figure(1)
    fig.clf()
    subplots = []
    subplots.append(fig.add_subplot(1, 2, 1))
    plt.imshow(edges)
    plt.title('Edge image')

    # display original image
    subplots.append(fig.add_subplot(1, 2, 2))
    plt.imshow(image)

    # draw the detected circle
    for i in range(len(radius)):
        blob_circ = plt_patches.Circle((center[i][1], center[i][0]), radius[i], fill=False, ec='red')
        plt.gca().add_patch(blob_circ)

    #   Fix axis distortion:
    plt.axis('image')

    if output:
        plt.savefig(output)

    plt.draw()
    plt.show()

    return


def _topNCircles(acc, radii, n):
    signal_list = []
    signal_max_positions = []
    radius_list = []
    for i, r in enumerate(radii):
        signal_max_positions.append(np.unravel_index(acc[i].argmax(), acc[i].shape))
        signal_list.append(acc[i].max())
        radius_list.append(r)

        """add second best"""
        center_gap = 25
        max_pos = np.unravel_index(acc[i].argmax(), acc[i].shape)


        acc[i, max_pos[0] - center_gap: max_pos[0] + center_gap, max_pos[1] - center_gap: max_pos[1] + center_gap] = 0

        # print(acc[i][max_pos[0] - center_gap: max_pos[0] + center_gap][max_pos[1] - center_gap: max_pos[1] + center_gap])
        signal_max_positions.append(np.unravel_index(acc[i].argmax(), acc[i].shape))
        signal_list.append(acc[i].max())
        radius_list.append(r)

        # use the radius to normalize
        signal_list[i] = signal_list[i] / np.sqrt(float(r))

    rad_apart = 0
    circle_pos = []
    radius = []

    for i in range(n):
        max_index = signal_list.index(max(signal_list))
        circle_pos.append(signal_max_positions[max_index])
        radius.append(radius_list[max_index])

        # for j in range(max(max_index-rad_apart,1),min(max_index+rad_apart,len(signal_list)-1)):
        #     signal_list[j] = 0

        signal_list[max_index] = 0

        # if signal > threshold_signal:
        #     if len(maxima) < n:
        #     else:
        #         max_signal = signal
        #     (circle_y, circle_x) = max_positions[i]
        #     radius = r
        #  print("Maximum signal for radius %d: %d %s, normal signal: %f" % (r, maxima[i], max_positions[i], signal))

    # Identify maximum. Note: the values come back as index, row, column
    #    max_index, circle_y, circle_x = np.unravel_index(acc.argmax(), acc.shape)

    return circle_pos, radius  # radii[max_index]


def DetectCircleFromFile(filename, circle_num, max_rad, min_rad, show_result=False):
    image = plt.imread(filename)
    center, radius = DetectCircle(image, circle_num, max_rad, min_rad, True, show_result)
    return center, radius


def DetectCircle(image, circle_num, max_rad, min_rad, preprocess=False, show_result=False):
    if preprocess:
        if image.ndim > 2:
            image = np.mean(image, axis=2)
        # print("Image size: ", image.shape)

        # noise reduction
        image = filters.gaussian_filter(image, 2)

        # edges and density
        edges = _detectEdges(image, _EDGE_THRESHOLD)
        edge_list = np.array(edges.nonzero())
        density = float(edge_list[0].size) / edges.size
        # print("Signal density:", density)
        # if density > 0.25:
        #     print("High density, consider more preprocessing")

    _MIN_RADIUS = min_rad
    _MAX_RADIUS = max_rad
    _RADIUS_STEP = 1
    # create kernels and detect circle
    radii = np.arange(_MIN_RADIUS, _MAX_RADIUS, _RADIUS_STEP)
    acc = _detectCircles(edges, radii, _ANNULUS_WIDTH)
    center, radius = _topNCircles(acc, radii, circle_num)
    # print("Circle detected at ", center, radius)

    if show_result:
        _displayResults(image, edges, center, radius)

    return center, radius


def _run(filename):
    return DetectCircleFromFile(filename)


def main(file_path, circle_num=4, max_rad=120, min_rad=20, ada_version=False):
    #    ## TODO: test cases, especially for iterative approach
    #    radii = np.arange(_MIN_RADIUS, _MAX_RADIUS, _RADIUS_STEP)
    #    image, edges = _initialize('mri.png')
    #    acc = _detectCircles(edges, radii, _ANNULUS_WIDTH)
    #    center, radius = _topNCircles(acc, radii, 1)
    #    print "Circle detected at ", center, radius
    #    _displayResults(image, edges, acc, radii, "mri-result.png")
    #    return
    if ada_version is False:
        center, radius = DetectCircleFromFile(file_path, circle_num, max_rad, min_rad, False)
        circle_det = []

        for i in range(len(radius)):
            circle_det.append(Circle(center[i][0], center[i][1], radius[i]))
        return circle_det
    else:
        center, radii = DetectCircle(file_path, circle_num, max_rad, min_rad, True)
        return center, radii


#    DetectCircleFromVideo("/home/vinnie/workspace/Intelligent-Artifacts/AMID/data/MRI_SAX_AVI/321/(MAIN)04052012-173005.avi")

if __name__ == "__main__":
    if (len(sys.argv) > 1):
        _run(sys.argv[1])
    else:
        # path = "C:/Users/liboyan/Desktop/hole1.jpg"
        path = "C:/Users/liboyan/Desktop/Industrial_PCB_Image_Dataset/Industrial PCB Image Dataset/1.bmp"
        main(path)
