import cv2
import random
import math
import numpy as np
import copy



def rand_rotate(img_src, angle_min=0, angle_max=45):
    assert -90 <= angle_min <= 90
    assert -90 <= angle_max <= 90
    assert angle_min < angle_max

    old_w = img_src.shape[1]
    old_h = img_src.shape[0]

    angle_degrees_origin = random.randint(angle_min, angle_max)
    angle_degrees_copy = abs(angle_degrees_origin)

    angle_radians = math.pi * angle_degrees_copy / 180.
    new_h = old_w * math.sin(angle_radians) + old_h * math.cos(angle_radians)
    new_w = old_w * math.cos(angle_radians) + old_h * math.sin(angle_radians)
    new_w = int(new_w)
    new_h = int(new_h)
    new_w_dim = max(new_w, old_w)

    translation_x = (new_w_dim - old_w) // 2
    translation_y = abs((new_h - old_h) / 2)

    translation_matrix = np.array([
        [1, 0, translation_x],
        [0, 1, translation_y]
    ], dtype=np.float32)
    img_translated = cv2.warpAffine(img_src, translation_matrix, (new_w_dim, new_h), borderValue=(255, 255, 255))

    center = (old_w // 2, new_h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle_degrees_origin, 1)
    img_dst = cv2.warpAffine(img_translated, rot_mat, (new_w_dim, new_h), borderValue=(255, 255, 255))

    if new_w_dim > new_w:
        border = int((new_w_dim - new_w) // 2)
        img_dst = img_dst[:, border:-border]

    return img_dst


def unwarp(img):
    # '/home/denis/Desktop/sudoku.png'
    pts1 = np.float32([[74, 89], [494, 71], [38, 515], [521, 519]])
    pts2 = np.float32([[0, 0], [500, 0], [0, 500], [500, 500]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (500, 500))
    return dst

def rand_darkening(img_src, margin_min=10, margin_max=100):
    shadow = random.uniform(margin_min, margin_max)
    alpha = (255 - shadow) / 255
    gamma = 0
    img_dst = cv2.addWeighted(img_src, alpha, img_src, 0, gamma)

    return img_dst

def rand_warp_tilt_as_starwars(img_src, margin_min=0.001, margin_max=0.1):
    w = img_src.shape[1]
    h = img_src.shape[0]

    begin_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    border_relative = random.uniform(margin_min, margin_max)
    up_left = [w * border_relative, h * border_relative]
    up_right = [w * (1 - border_relative), h * border_relative]
    down_left = [0, h]
    down_right = [w, h]
    end_points = np.float32([up_left, up_right, down_left, down_right])

    transform_mat = cv2.getPerspectiveTransform(begin_points, end_points)
    img_dst = cv2.warpPerspective(img_src, transform_mat, (w, h), borderValue=(255, 255, 255))
    img_dst = img_dst[int(border_relative * h):, :]

    return img_dst


def rand_warp_tilt_right(img_src, margin_min=0.001, margin_max=0.1):
    w = img_src.shape[1]
    h = img_src.shape[0]

    begin_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    border_relative = random.uniform(margin_min, margin_max)
    up_left = [0, 0]
    up_right = [w * (1 - border_relative), h * border_relative]
    down_left = [0, h]
    down_right = [w * (1 - border_relative), h * ( 1- border_relative)]
    end_points = np.float32([up_left, up_right, down_left, down_right])

    transform_mat = cv2.getPerspectiveTransform(begin_points, end_points)
    img_dst = cv2.warpPerspective(img_src, transform_mat, (w, h), borderValue=(255, 255, 255))
    img_dst = img_dst[:, :int(w * ( 1 - border_relative))]

    return img_dst


def rand_warp_tilt_left(img_src, margin_min=0.001, margin_max=0.1):
    w = img_src.shape[1]
    h = img_src.shape[0]

    begin_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    border_relative = random.uniform(margin_min, margin_max)
    up_left = [w * border_relative, h * border_relative]
    up_right = [w, 0]
    down_left = [w * border_relative, h * (1 - border_relative)]
    down_right = [w, h]

    end_points = np.float32([up_left, up_right, down_left, down_right])

    transform_mat = cv2.getPerspectiveTransform(begin_points, end_points)
    img_dst = cv2.warpPerspective(img_src, transform_mat, (w, h), borderValue=(255, 255, 255))
    img_dst = img_dst[:, int(w * border_relative):]

    return img_dst


def rand_rotate_and_warp_tilt_as_starwars(img_src, rotate_options, warp_options):
    img_rotated = rand_rotate(img_src, *rotate_options)
    img_warped = rand_warp_tilt_as_starwars(img_rotated, *warp_options)
    return img_warped


def rand_rotate_and_warp_tilt_left(img_src, rotate_options, warp_options):
    img_rotated = rand_rotate(img_src, *rotate_options)
    img_warped = rand_warp_tilt_left(img_rotated, *warp_options)
    return img_warped


def rand_rotate_and_warp_tilt_right(img_src, rotate_options, warp_options):
    img_rotated = rand_rotate(img_src, *rotate_options)
    img_warped = rand_warp_tilt_right(img_rotated, *warp_options)
    return img_warped


def rand_warp_tilt_as_starwars_and_warp_tilt_left(img_src, rotate_options, warp_options):
    img_warp_tilt_as_starwars = rand_warp_tilt_as_starwars(img_src, *rotate_options)
    img_warp_tilt_left = rand_warp_tilt_left(img_warp_tilt_as_starwars, *warp_options)
    return img_warp_tilt_left


def rand_warp_tilt_as_starwars_and_warp_tilt_right(img_src, rotate_options, warp_options):
    img_warp_tilt_as_starwars = rand_warp_tilt_as_starwars(img_src, *rotate_options)
    img_warp_tilt_right = rand_warp_tilt_right(img_warp_tilt_as_starwars, *warp_options)
    return img_warp_tilt_right


def rand_blur(img):
    prob = random.random()
    if prob > 0.75:
        blurred_img = cv2.blur(img, (3, 3))
        return blurred_img

    if prob < 0.25:
        img_detail_enhanced = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
        return img_detail_enhanced

    return copy.deepcopy(img)


if __name__ == '__main__':
    # img = cv2.imread('/home/denis/Desktop/sample/8.jpg')
    img = cv2.imread('/home/kate/Desktop/229239.jpg')
    scale_factor = 4

    width = int(img.shape[1] * scale_factor)
    height = int(img.shape[0] * scale_factor)
    dsize = (width, height)

    # resize image
    img = cv2.resize(img, dsize)


    cv2.imshow('origin', img)
    # img = rand_rotate(img, -5, 5)
    # img = rand_warp_tilt_as_starwars(img, 0.01, 0.15)
    # img = rand_rotate_and_warp_tilt_as_starwars(img, (0, 10), (0.01, 0.1))
    # img = rand_warp_tilt_left(img, margin_min=0.01, margin_max=0.2)
    # img = rand_warp_tilt_right(img, margin_min=0.2, margin_max=0.2)
    # img = rand_warp_tilt_as_starwars_and_warp_tilt_right(img, (0.01, 0.15), (0.01, 0.2))
    # img = rand_warp_tilt_as_starwars_and_warp_tilt_left(img, (0.01, 0.15), (0.01, 0.2))
    # img = rand_warp_tilt_as_starwars_and_warp_tilt_left(img, (0, 0.1), (0, 0.3))
    # img = rand_rotate_and_warp_tilt_left(img, (-15, -10), (0.1, 0.2))
    img =  rand_darkening(img, 10, 10)
    cv2.imshow('augmented', img)
    cv2.waitKey(0)
