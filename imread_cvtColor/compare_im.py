"""
Test scripts
"""


def compare(im1, im2):
    assert im1.shape == im2.shape

    print(type(im1), im1.dtype)
    print(type(im2), im2.dtype)

    diff = np.where(im1 != im2)
    print(diff)
    all_num = im1.shape[0] * im1.shape[1]
    print("Diff num:{}/{}={}".format(
        len(diff[0]), all_num, len(diff[0]) / all_num))

    print(im1[diff])
    print(im2[diff])


def compare_color(im1, im2):
    assert im1.shape == im2.shape

    b1, g1, r1 = cv2.split(im1)
    b2, g2, r2 = cv2.split(im2)

    print("Diff in blue")
    compare(b1, b2)

    print("Diff in green")
    compare(g1, g2)

    print("Diff in red")
    compare(r1, r2)


if __name__ == "__main__":

    import cv2
    import numpy as np

    import matplotlib.cbook as cbook
    im_file = cbook.get_sample_data(
        'grace_hopper.png').name  # 512*600, shape (600, 512, 3)

    im_color_direct = cv2.imread(im_file)
    im_gary = cv2.cvtColor(im_color_direct, cv2.COLOR_BGR2GRAY)
    im_color_convert_back = cv2.cvtColor(im_gary, cv2.COLOR_GRAY2BGR)
    """
    cv2.imshow("im_color_direct", im_color_direct)
    cv2.waitKey(0)
    cv2.imshow("im_gary", im_gary)
    cv2.waitKey(0)
    """

    im_gray_direct = cv2.imread(im_file, 0)
    im_color = cv2.cvtColor(im_gray_direct, cv2.COLOR_GRAY2BGR)

    print("im_gary, im_gray_direct:")
    compare(im_gary, im_gray_direct)

    print("\n\nim_color, im_color_direct")
    compare_color(im_color, im_color_direct)

    print("\n\nim_color_direct, im_color_convert_back")
    compare_color(im_color_direct, im_color_convert_back)
