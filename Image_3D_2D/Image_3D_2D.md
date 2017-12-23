Manipulating 3D points and 2D planes can be accomplished via hand-written codes or seeking help from libraries. And I prefer the better one.

> Q: How to convert RGB+D images into another perspective?

[How to align RGB and Depth image of Kinect in OpenCV?](https://stackoverflow.com/questions/21849512/how-to-align-rgb-and-depth-image-of-kinect-in-opencv) showed something we might consider: `... since you need to handle occlusions (by keeping only the minimum depth seen by each pixel) and image interpolation (since in general, the projected 3D points won't be associated with integer pixel coordinates in the RGB image) ... `

And maybe the combination of `reprojectImageTo3D()` and `projectPoints()` can be used for this purpose.

Even the [`CreateFromPointCloud` from PCL library](https://github.com/udacity/RoboND-Perception-Exercises/blob/master/python-pcl/pcl/pxi/Common/RangeImage/RangeImagePlanar_172.pxi) looks nice.

At last, [`warpFrame()` seems just meet this requirement](https://docs.opencv.org/3.3.0/d2/d3a/group__rgbd.html#gac0db6aeba01fa17ec2c69694497926f0), no more, no less.

However, there will be holes in the warped images. And as for the depth image, `registerDepth()` provide similar result (`warpFrame` call it ~underhood~). But it is not exported to python binding -.-


