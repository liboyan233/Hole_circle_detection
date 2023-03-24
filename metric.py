import numpy as np


class Circle:
    def __init__(self, x=0, y=0, rad=0):
        """ Create a new point at the origin """
        self.x = x
        self.y = y
        self.radius = rad


def metric_VRIc(circle_det_list, circle_gt_list):
    beta = 0.5
    if not circle_det_list:
        return 0
    VRIc = beta*Cd(circle_det_list, circle_gt_list) + (1-beta)*Cf(circle_det_list, circle_gt_list)
    return VRIc


def Cd(circle_det_list, circle_gt_list):
    cd = 0
    N_gt = len(circle_gt_list)
    for circle_gt in circle_gt_list:
        ov_list = []
        for circle_det in circle_det_list:
            ov = Ov(circle_det, circle_gt)
            ov_list.append(ov)
        if max(ov_list) >= 0.5:
            cd += max(ov_list)/N_gt
    return cd


def Cf(circle_det_list, circle_gt_list):
    cf = 1
    N_d = len(circle_det_list)
    for circle_det in circle_det_list:
        ov_list2 = []
        for circle_gt in circle_gt_list:
            ov = Ov(circle_det, circle_gt)
            ov_list2.append(ov)
        if max(ov_list2) < 0.5:
            cf -= max(ov_list2)/N_d
    return cf


def Ov(circle1, circle2):

    radius1 = circle1.radius
    radius2 = circle2.radius

    if radius1 >= radius2:
        max_circle = np.pi * radius1**2
    else:
        max_circle = np.pi * radius2**2

    center_dist = distance(circle1, circle2)

    if center_dist >= radius1 + radius2:
        return 0
    elif radius1 + center_dist <= radius2:
        return np.pi * radius1 ** 2 / max_circle
    elif radius2 + center_dist <= radius1:
        return np.pi * radius2 ** 2 / max_circle
    else:
        ang1 = 2 * np.arccos((radius2**2 + center_dist**2 - radius1**2)/(2*radius2*center_dist))  # arccos range: 0-pi
        ang2 = 2 * np.arccos((radius1**2 + center_dist**2 - radius2**2)/(2*radius1*center_dist))
        return (arc_area(radius2, ang1) + arc_area(radius1, ang2))/max_circle


def distance(c1, c2):
    dist_x = c1.x - c2.x
    dist_y = c1.y - c2.y
    return (dist_y**2 + dist_x**2)**0.5


def arc_area(radius, angle):

    arc = radius ** 2 * angle / 2
    if np.pi*2 >= angle > np.pi:
        arc_total = arc + radius **2 * np.sin(np.pi*2 - angle)
    elif np.pi >= angle >= 0:
        arc_total = arc - radius **2 * np.sin(angle)
    else:
        my_error = ValueError(str(angle) + "is not a valid angle")
        raise my_error
    return arc_total

