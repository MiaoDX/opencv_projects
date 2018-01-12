# The x,y and row,col in OpenCV image

Accessing image value of particular pixel can be a little surprising.

![Answer to OpenCV Point(x,y) represent (column,row) or (row,column)](https://stackoverflow.com/a/25644503/7067150)

```
0/0---column--->
 |
 |
row
 |
 |
 v
```

```
0/0---X--->
 |
 |
 Y
 |
 |
 v
```

Tweak a while we can understand the relationship, [the doc of OpenCV describes equally well](https://docs.opencv.org/2.4/doc/user_guide/ug_mat.html):

> Accessing pixel intensity values <br>
In order to get pixel intensity value, you have to know the type of an image and the number of channels. Here is an example for a single channel grey scale image (type 8UC1) and pixel coordinates x and y:
<br>
`Scalar intensity = img.at<uchar>(y, x);`    
intensity.val[0] contains a value from 0 to 255. Note the ordering of x and y. Since in OpenCV images are represented by the same structure as matrices, we use the same convention for both cases - the 0-based row index (or y-coordinate) goes first and the 0-based column index (or x-coordinate) follows it. Alternatively, you can use the following notation:
<br>
Scalar intensity = img.at<uchar>(Point(x, y));

This difference recalls me when dealing with the depth images captured from rgbd sensors, we get the keypoints first, and want to retrieve the depth value, we should also take the data type into consideration (most depth images are saved as 16bit png format):

```
uint16_t d = depthMap.at<uint16_t>(y, x);
uint16_t d = depthMap.ptr<uint16_t>(y)[x];

uint16_t d = depthMap.ptr<uint16_t>(Point(x, y));


uint16_t d = depthMap.at<uint16_t>(int(y), int(x)); // BETTER
```

And in most compilers the `uint16_t` is the same as `ushort`, but I personally prefer the former one since the image is also `16bit`. Note the last `int()` operation, the depth image should be accessed with int values in most cases.

## In python OpenCV

The statement still holds, for example an (640*480) RGBA png image will have shape (480, 640, 4). And for feature points detection, the `kp.pt[0]` and `kp.pt[1]` will be x and y respectively, so, to grab the corresponding pixel value (especially when dealing with depth image), we should use `im[kp.pt[1], kp.pt[0]]`. And this can be a little difficult to remember all the time (error throw by the program may help you recognize it -.-).

