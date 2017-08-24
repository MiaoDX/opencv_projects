To create `real-time` application, it is more than crucial to use the GPU mode computing, especially for the feature detector, descriptor extractor and matching. And with opencv cuda module, it's very easy and straight-forward, the lack of GPU SIFT can be a disappointment, but, we can not have it all, right?

There are two examples suitable for OpenCV 3.3.0 (not so sure if applies to other versions since the API changes among versions).

* [`CMakelists.txt`](CMakelists.txt) file is copied from [gpu_testing](https://github.com/ariandyy/gpu_testing)
* [surf_keypoint_matcher.cpp](surf_keypoint_matcher.cpp) is the example code of [opencv official](https://github.com/opencv/opencv/blob/3.3.0/samples/gpu/surf_keypoint_matcher.cpp)
* [surf_orb_cuda_cv330.cpp](surf_orb_cuda_cv330.cpp) is almost we all need.