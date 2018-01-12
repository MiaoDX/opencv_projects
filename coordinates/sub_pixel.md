There are various functions in OpenCV accepts pixel coordinates as floating point numbers. Even the most widely used pixel access of Mat value, especially with feature points, lots of implements just use the points to retrieve RGB values while many of others do the rounding operation for Depth info. Seems no constant usage.

[Absent documentation for sub-pixel coordinate system](https://github.com/opencv/opencv/issues/10130)

[Kirill888/cv2-test-remap.ipynb](https://gist.github.com/Kirill888/0861c2508d41766a0145682e330ff8f5)

[Nearest neighbor interpolation does not give expected results #9096](https://github.com/opencv/opencv/issues/9096)

# The influence on accessing depth

There can be some influences on the depth value retrieve, see a demo example:

``` cpp
#include "opencv2/core.hpp"
#include <iostream>

using namespace cv;
using namespace std;


int main ()
{
    Mat R = Mat ( 3, 4, CV_16UC1 );
    randu ( R, Scalar::all ( 0 ), Scalar::all ( 255 ) );

    cout << R << endl;

    uint16_t d1 = R.at<uint16_t> ( 1, 2 );

    uint16_t d2 = R.at<uint16_t> ( Point2f ( 2, 1 ) );

    uint16_t d3 = R.at<uint16_t> ( 1.2, 2.99999 );

    uint16_t d4 = R.at<uint16_t> ( Point2f ( 2.6, 0.9 ) );

    cout << "d1:" << d1 << "\td2:" << d2 << "\td3:" << d3 << "\td4:" << d4 << endl;

    system ( "pause" );
    return 0;
}
```

The result is:

``` vim
[91, 2, 79, 179;
 52, 205, 236, 8;
 181, 239, 26, 248]
d1:236  d2:236  d3:236  d4:8
```

The first three makes no big different, what's interesting is the last one, it access Point(3, 1), i.e. rounding the float values into int, not `floor` operation. Anyway, when we use depth information, the depth is likely to be similar in a region, so the impact can be not so large.

## Takeaway

It should be proper to use `round()` when we access depth info in python code where there will be no 