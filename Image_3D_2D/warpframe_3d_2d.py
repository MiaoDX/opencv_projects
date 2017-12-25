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

if __name__ == '__main__':

    assert imutils.is_cv3()


    # im1_file = '0_0_0_0_0_0.png'
    # im1_d_file = '0_0_0_0_0_0_depth.png'
    # im2_file = '0_0_0_5_10_15.png'
    # im2_d_file = '0_0_0_5_10_15_depth.png'
    im1_file = 'zed_im/ref_init.png'
    im1_d_file = 'zed_im/ref_init_depth.png'
    im2_file = 'zed_im/20171225_231831_0_lit.png'
    im2_d_file = 'zed_im/20171225_231831_0_depth.png'


    im1 = cv2.imread(im1_file)
    im1_d = cv2.imread(im1_d_file, cv2.IMREAD_UNCHANGED)
    assert im1 is not None and im1_d is not None
    # im1_d = np.float32(im1_d)
    # im1_d /= 1000.0

    im2 = cv2.imread(im2_file)
    im2_d = cv2.imread(im2_d_file, cv2.IMREAD_UNCHANGED)
    assert im2 is not None and im2_d is not None

    # R2 = zyx2R(5, 10, 15) # note the sequence is not like what we want, doc said `Rt The transformation that will be applied to the 3d points computed from the depth`
    # t2 = [-0.0, 0, 0]
    # R2 = np.eye(3)
    # t2 = [-0.478, -0.02, -0.13]
    # Rt2 = Rt44(R2, t2)

    R2 = zyx2R(-2.79, 9.94, 3.32)
    t2 = [0.478, 0.02, 0.13]
    Rt2 = Rt44_inv(R2, t2)


    K = np.array([[320, 0, 320], [0, 320, 240], [0, 0, 1]]).astype(np.float32)

    # warpFrame(image, depth, mask, Rt, cameraMatrix, distCoeff[, warpedImage[, warpedDepth[, warpedMask]]]) -> warpedImage, warpedDepth, warpedMask
    warpedImage, warpedDepth, _ = cv2.rgbd.warpFrame(im1, im1_d, None, Rt2, K, None)
    cv2.imshow('RGB', warpedImage)

    plt.imshow(warpedDepth-im2_d/1000.0)
    plt.show()

    cv2.waitKey(0)


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