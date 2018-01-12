There are various coordinates out there, left/right-handed, image/camera/world coordinate. Combined with rotation/translation, things get worse. Let us clarify these one by one. Note, we try to be constant with OpenCV.

# The coordinates

![Coordinate in OpenCV and OpenMVG](pics/pinholeCamera.png)

Image from [OpenMVG doc](http://openmvg.readthedocs.io/en/latest/openMVG/cameras/cameras/#pinhole-camera-model).

![Coordinate systems of Ubitrack, OpenCV and OpenGL](pics/coor_cv_gl.png).

Image from [Note for coordinate system conversion](http://campar.in.tum.de/twiki/pub/Main/YutaItoh/coordinate_system_summary2.pdf).

# NOTES

* [The x,y and row,col in OpenCV image](xy_col_row.md)
* [The sub-pixel phenomenon](sub_pixel.md)
