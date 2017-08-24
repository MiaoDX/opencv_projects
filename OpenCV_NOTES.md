Some notes of OpenCV

## [Mat, vector<point2f>，Iplimage等等常见类型转换](http://blog.csdn.net/foreverhehe716/article/details/6749175)

Seems not so right.

``` cpp
vector<Point2f> ptsa = Mat_<Point2f>(x1s);
```



## Format printf

``` vi
size_t x = ...;
ssize_t y = ...;
printf("%zu\n", x);  // prints as unsigned decimal
printf("%zx\n", x);  // prints as hex
printf("%zd\n", y);  // prints as signed decimal


%llu // for unsigned __int64
```