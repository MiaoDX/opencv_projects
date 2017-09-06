# OpenCV projects

Some notes and example codes and repositories' links of projects with OpenCV, I will try my best to make the codes works for both OpenCV 2.x (2.4.11) and OpenCV 3.x (3.3.0), and some wrote with python.

The projects in this repo are relative small, and there are some notes:

* [imread_cvtColor](imread_cvtColor)
    - shows the difference of reading image directly and read in one format and convert it with program, the `GRAY` mode, to be specific.

* [opencv_gpu](opencv_gpu)
    - Simple example code of using cuda support feature detectors and descriptors with OpenCV 3.3.0 -- `SURF_CUDA` and `cuda::ORB`, the code is up-to-date by far (201709), but the APIs change dramatically, so great chances are that this can be unsuitable for other version.

* [real_time_pose_estimation](real_time_pose_estimation)
    - Nothing but a copy from official code, add separate CMakeLists to make it run.



And there are some other projects, which are somewhat self-contained and much more bigger, so they are stored and maintained in other places:


* [pose-estimation](https://github.com/MiaoDX/pose_estimation) 
    - where I do lots of experiments on feature detection and camera pose calibration, codes in both C++ and python, give it a look! :)



***

Good Luck & Have Fun.