There are some differences between `imread` directly and use `cvtColor` when we are using OpenCV.

I recognize this difference when I port a program (use `findEssentialMat` and `recoverPose` to estimate the relative pose between images) into one `OOP` version, and I wanted to store both the color and gray image in the class for later display, but the answers of R,t are totally(/somewhat) different. I went through the code once and once again, and it trace back to the part when calculating the matched keypoints between the images, one answer is 1233 matched points, the other is 1234. This difference is unbelievable for me, since the code before that is almost the same, and at last I figured out the only difference is the way I got the gray image:

``` python
# The serial version
im_gray = cv2.imread(im_name, 0)

# The OOP version
im_color = cv2.imread()
im_gray = cv2.cvtColor(im_color, cv2.COLOR_BGR2GRAY)
```

And the underlying operation details makes the answers differ, though the final differences may not be too large, it dose spend me a lot time to figure out.

And in the [compare_im.py](compare_im.py), I provide some scripts to verify my *finding*.


So, one problem show up: **Which image should we use for our computing?**, well, I am not so sure of that right now, will do some experiments and revisit this.