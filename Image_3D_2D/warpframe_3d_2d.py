"""
There are some coordinate problems as the codes shows
One explanation is that it is [`Rt The transformation that will be applied to the 3d points computed from the depth`](https://docs.opencv.org/3.3.0/d2/d3a/group__rgbd.html)
The other possible issue is [In the `depthTo3d` method, The coordinate system is x pointing left, y down and z away from the camera](https://docs.opencv.org/3.3.0/d2/d3a/group__rgbd.html)

These two comments are conflict. E.g. in our normal coordinate, move camera x=40 (move right), we got `40_0_0_0_0_0.png`
If the Rt is for 3d points, then will be the inverse of camera, so, the t=[-0.4, 0, 0]
And if the strange coordinate, then together with move 3d points, the t=[0.4, 0, 0]

Result show that the former is right, so we can just ignore the coordinate description.
"""

import cv2
import imutils
import numpy as np
import math
import transforms3d
import matplotlib.pyplot as plt

def Rt44(R, t):
    R = np.array(R).reshape(3, 3)
    t = np.array(t)
    Rt44 = np.eye(4)
    Rt44[:3,:3] = R
    Rt44[:3, 3] = t
    print(Rt44)

    return Rt44

def Rt44_inv(R, t):
    Rt = Rt44(R, t)
    Rt_inv = np.linalg.inv(Rt)
    print(Rt_inv)
    return Rt_inv

def zyx2R(z_degree, y_degree, x_degree):
    z_rad = math.radians(z_degree)
    y_rad = math.radians(y_degree)
    x_rad = math.radians(x_degree)

    R = transforms3d.euler.euler2mat(z_rad, y_rad, x_rad, 'rzyx')
    return R


def warp_and_subtract(im1_file, im1_d_file, im2_file, im2_d_file, Rt1_to_2_cam):
    im1 = cv2.imread(im1_file)
    im1_d = cv2.imread(im1_d_file, cv2.IMREAD_UNCHANGED)
    assert im1 is not None and im1_d is not None
    # im1_d = np.float32(im1_d)
    # im1_d /= 1000.0

    im2 = cv2.imread(im2_file)
    im2_d = cv2.imread(im2_d_file, cv2.IMREAD_UNCHANGED)
    assert im2 is not None and im2_d is not None

    # R = zyx2R(-2.79, 9.94, 3.32)
    # t = [0.478, 0.02, 0.13]
    # Rt = Rt44_inv(R, t)

    # Rt_obj = np.linalg.inv(Rt1_to_2_cam)
    Rt_obj = Rt1_to_2_cam

    K = np.array([[320, 0, 320], [0, 320, 240], [0, 0, 1]]).astype(np.float32)

    # warpFrame(image, depth, mask, Rt, cameraMatrix, distCoeff[, warpedImage[, warpedDepth[, warpedMask]]]) -> warpedImage, warpedDepth, warpedMask
    im_warp, im_warp_d, _ = cv2.rgbd.warpFrame(im1, im1_d, None, Rt_obj, K, None)



    # cv2.imshow('RGB', im_warp)
    # plt.imshow(im_warp_d*1000.0-im2_d)
    # plt.show()
    # cv2.waitKey(0)

    print(im_warp.shape)
    print(im_warp_d.shape)

    return im_warp, im_warp_d

    # bi_f = cv2.bilateralFilter(warpedImage, 5, 80, 80)
    # # bi_f = cv2.adaptiveBilateralFilter (warpedImage, 15, 80)
    # cv2.imshow('RGB bilateralFilter', bi_f)
    # plt.imshow(warpedImage-bi_f)
    # plt.show()
    # cv2.waitKey(0)


    # gauss_b = cv2.GaussianBlur(warpedImage, (5,5), 0)
    # cv2.imshow('RGB bilateralFilter', gauss_b)
    # plt.imshow(warpedImage - gauss_b)
    # plt.show()
    # cv2.waitKey(0)


def load_Rt_and_file_prefix(j_file='zed_im/1.json'):
    import json_tricks
    with open(j_file, 'r') as f:
        info = json_tricks.load(f)

    R = info['R']
    t = np.array(info['t']).flatten()/1000.0 # NOTE

    Rt = Rt44(R, t)

    cur_file = info['im_cur_file'].split('/')[-1] # 20171225_231831_0_lit.png
    cur_file_prefix = cur_file[:-7] # 20171225_231831_0_
    print(cur_file_prefix)

    return Rt, cur_file_prefix


if __name__ == '__main__':

    assert imutils.is_cv3()


    # im1_file = '0_0_0_0_0_0.png'
    # im1_d_file = '0_0_0_0_0_0_depth.png'
    # im2_file = '0_0_0_5_10_15.png'
    # im2_d_file = '0_0_0_5_10_15_depth.png'
    im_ref = 'zed_im/ref_init.png'
    im_ref_d = 'zed_im/ref_init_depth.png'

    # im2_file = 'zed_im/20171225_231831_0_lit.png'
    # im2_d_file = 'zed_im/20171225_231831_0_depth.png'

    for i in range(1, 13):
        j_file = 'zed_im/'+str(i)+'.json'

        # input('press to continue')

        Rt, cur_file_prefix = load_Rt_and_file_prefix(j_file)

        im2_file = 'zed_im/' + cur_file_prefix + 'lit.png'
        im2_d_file = 'zed_im/' + cur_file_prefix + 'depth.png'

        im_warp_file, im_warp_d_file = 'zed_im/' + cur_file_prefix + 'lit_warp.png', 'zed_im/' + cur_file_prefix + 'depth_warp.png'


        im_warp, im_warp_d = warp_and_subtract(im2_file, im2_d_file, im_ref, im_ref_d, Rt)
        im_warp_d *= 1000.0
        im_warp_d = im_warp_d.astype(np.uint16)


        cv2.imwrite(im_warp_file, im_warp)
        cv2.imwrite(im_warp_d_file, im_warp_d)


