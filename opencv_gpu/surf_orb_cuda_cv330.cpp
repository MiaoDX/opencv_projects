/*
 * Usage: --left H:/projects/SLAM/python_code/dataset/our/trajs/1.bmp --right H:/projects/SLAM/python_code/dataset/our/trajs/2.bmp
 */
#include <iostream>
#include <vector>
#include <string>
#include <algorithm> // toupper
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#ifdef _CV_VERSION_3
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#else
#include <opencv2/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/nonfree/gpu.hpp>
#endif


using namespace std;
using namespace cv;
using namespace cv::cuda;

static void help()
{
    cout << "\nThis program demonstrates using SURF_CUDA features detector, descriptor extractor and BruteForceMatcher_CUDA" << endl;
    cout << "\nUsage:\n\tsurf_keypoint_matcher --left <image1> --right <image2>" << endl;
}

void _extract_keypoints_and_matches_surf ( const Mat& im1_gray, const Mat& im2_gray,
        vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2, vector<DMatch>& matches );
void _match_with_knnMatch_cuda ( const Ptr<cuda::DescriptorMatcher>& matcher, const GpuMat& des1, const GpuMat& des2,
                                 vector<DMatch>& matches, float minRatio = 0.7 );
void _match_with_NORM_HAMMING_cuda ( const Ptr<cuda::DescriptorMatcher>& matcher, const GpuMat& des1, const GpuMat& des2, vector<DMatch>& matches, double threshold_dis = 30.0 );
void _extract_keypoints_and_matches_orb ( const Mat& im1_gray, const Mat& im2_gray,
        vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2, vector<DMatch>& matches, const int nfeatures = 1000 );

int main(int argc, char* argv[])
{
    if (argc != 5) {
        help();
        return -1;
    }

    Mat im1_color, im2_color;

    for ( int i = 1; i < argc; ++i ) {
        if ( string ( argv[i] ) == "--left" ) {
            im1_color = imread ( argv[++i]);
            CV_Assert ( !im1_color.empty () );
        }
        else if ( string ( argv[i] ) == "--right" ) {
            im2_color = imread ( argv[++i] );
            CV_Assert ( !im2_color.empty () );
        }
        else if ( string ( argv[i] ) == "--help" ) {
            help ();
            return -1;
        }
    }

    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

    Mat im1_gray, im2_gray;
    cvtColor ( im1_color, im1_gray, CV_BGR2GRAY );
    cvtColor ( im2_color, im2_gray, CV_BGR2GRAY );

    vector<KeyPoint> kpts1, kpts2;
    vector<DMatch> matches;

    //_extract_keypoints_and_matches_surf ( im1_gray, im2_gray, kpts1, kpts2, matches );
    _extract_keypoints_and_matches_orb ( im1_gray, im2_gray, kpts1, kpts2, matches, 5000 );


    // drawing the results
    Mat img_matches;
    drawMatches(Mat(im1_color), kpts1, Mat(im2_color), kpts2, matches, img_matches);


    namedWindow ( "matches", WINDOW_NORMAL ); // avoid too large to show
    imshow("matches", img_matches);
    waitKey(0);

    //system ( "pause" );

    return 0;
}

void _extract_keypoints_and_matches_surf ( const Mat& im1_gray, const Mat& im2_gray,
        vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2, vector<DMatch>& matches)
{
    // Should add assert for gray images

    GpuMat img1, img2;
    img1.upload ( im1_gray );
    img2.upload ( im2_gray );

    SURF_CUDA surf;

    // detecting keypoints & computing descriptors
    GpuMat keypoints1GPU, keypoints2GPU;
    GpuMat descriptors1GPU, descriptors2GPU;
    surf ( img1, GpuMat (), keypoints1GPU, descriptors1GPU );
    surf ( img2, GpuMat (), keypoints2GPU, descriptors2GPU );

    cout << "FOUND " << keypoints1GPU.cols << " keypoints on first image" << endl;
    cout << "FOUND " << keypoints2GPU.cols << " keypoints on second image" << endl;

    // matching descriptors, use NORM_L2 for our OpenCV 2.x
    // Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher ( surf.defaultNorm () );
    // cout << "Norm, surf:" << surf.defaultNorm () << ", NORM_L2:" << NORM_L2 << endl;
    Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher ( NORM_L2 );


    _match_with_knnMatch_cuda ( matcher, descriptors1GPU, descriptors2GPU, matches );


    // downloading results
    surf.downloadKeypoints ( keypoints1GPU, keypoints1 );
    surf.downloadKeypoints ( keypoints2GPU, keypoints2 );
    // vector<float> descriptors1, descriptors2;
    // surf.downloadDescriptors ( descriptors1GPU, descriptors1 );
    //surf.downloadDescriptors ( descriptors2GPU, descriptors2 );
}



void _extract_keypoints_and_matches_orb ( const Mat& im1_gray, const Mat& im2_gray,
        vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2, vector<DMatch>& matches, const int nfeatures )
{
    // Should add assert for gray images

    GpuMat img1, img2;
    img1.upload ( im1_gray );
    img2.upload ( im2_gray );

    Ptr<cuda::ORB> orb = cuda::ORB::create ( nfeatures );


    // detecting keypoints & computing descriptors
    GpuMat keypoints1GPU, keypoints2GPU;
    GpuMat descriptors1GPU, descriptors2GPU;

    orb->detectAndComputeAsync ( img1, noArray (), keypoints1GPU, descriptors1GPU );
    orb->detectAndComputeAsync ( img2, noArray (), keypoints2GPU, descriptors2GPU );

    cout << "FOUND " << keypoints1GPU.cols << " keypoints on first image" << endl;
    cout << "FOUND " << keypoints2GPU.cols << " keypoints on second image" << endl;

    Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher ( NORM_HAMMING );
    _match_with_NORM_HAMMING_cuda ( matcher, descriptors1GPU, descriptors2GPU, matches );

    // downloading results
    orb->convert ( keypoints1GPU, keypoints1 );
    orb->convert ( keypoints2GPU, keypoints2 );
}


void _match_with_knnMatch_cuda ( const Ptr<cuda::DescriptorMatcher>& matcher, const GpuMat& des1, const GpuMat& des2,
                                 vector<DMatch>& matches, float minRatio)
{
    matches.clear ();

    const int k = 2;

    vector<vector<DMatch>> knnMatches;
    matcher->knnMatch ( des1, des2, knnMatches, k );

    for ( size_t i = 0; i < knnMatches.size (); i++ ) {
        const DMatch& bestMatch = knnMatches[i][0];
        const DMatch& betterMatch = knnMatches[i][1];

        float  distanceRatio = bestMatch.distance / betterMatch.distance;
        if ( distanceRatio < minRatio )
            matches.push_back ( bestMatch );
    }

    cout << "knnMatches:" << knnMatches.size () << " -> " << matches.size () << endl;
}

void _match_with_NORM_HAMMING_cuda ( const Ptr<cuda::DescriptorMatcher>& matcher, const GpuMat& des1, const GpuMat& des2, vector<DMatch>& matches, double threshold_dis )
{

    matches.clear ();

    vector<DMatch> all_matches;
    matcher->match ( des1, des2, all_matches );

    double min_dist = 10000, max_dist = 0;


    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < all_matches.size (); i++ ) {
        double dist = all_matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    cout << "-- Max dist:" << max_dist;
    cout << ". Min dist:" << min_dist << endl;

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < all_matches.size (); i++ ) {
        if ( all_matches[i].distance <= max ( 2 * min_dist, threshold_dis ) ) {
            matches.push_back ( all_matches[i] );
        }
    }

    cout << "NORM_HAMMING match:" << all_matches.size () << " -> " << matches.size () << endl;
}



/**
 * \brief Sort matches, copy from [offical code matchmethod_orb_akaze_brisk](https://github.com/opencv/opencv/blob/master/samples/cpp/matchmethod_orb_akaze_brisk.cpp)
 * \param matches
 * \param matches_sorted
 */
void _sort_matches(const vector<DMatch> matches, vector<DMatch> matches_sorted)
{

    matches_sorted.clear ();

    Mat index;
    int nbMatch = int ( matches.size () );
    Mat tab ( nbMatch, 1, CV_32F );
    for ( int i = 0; i<nbMatch; i++ ) {
        tab.at<float> ( i, 0 ) = matches[i].distance;
    }
    sortIdx ( tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING );

    for ( int i = 0; i<nbMatch; i++ ) {
        matches_sorted.push_back ( matches[index.at<int> ( i, 0 )] );
    }
}