//
// Created by sicong on 25/11/17.
//

#ifndef CERES_CUSTOMBUNDLE_TRIANGULATION_H
#define CERES_CUSTOMBUNDLE_TRIANGULATION_H

#endif //CERES_CUSTOMBUNDLE_TRIANGULATION_H
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>

#include "tools/random.h"
#include "tools/rotation.h"
#include "tools/random.h"
#include "projection.h"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>


typedef Eigen::Map<Eigen::VectorXd> VectorRef;
typedef Eigen::Map<const Eigen::VectorXd> ConstVectorRef;

using namespace Eigen;
using namespace cv;
using namespace std;

void triangulation (
        const vector<Point2f>& point_1,
        const vector<Point2f>& point_2,
        const Mat& R, const Mat& t, const Mat& K,
        vector<Point3f>& points,vector<pair <pixel*, pixel*>> matchpairlist
);

Point2f pixel2cam ( const Point2f& p, const Mat& K );

void triangulation (
        const vector< Point2f >& pts2d_1,
        const vector< Point2f >& pts2d_2,
        const Mat& R, const Mat& t, const Mat& K,
        vector< Point3f >& points,vector<pair <pixel*, pixel*>> matchpairlist )
{
    Mat T1 = (Mat_<float> (3,4) <<
                                1,0,0,0,
            0,1,0,0,
            0,0,1,0);
    Mat T2 = (Mat_<float> (3,4) <<
                                R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0,0),
            R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1,0),
            R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2,0)
    );

    //Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    vector<Point2f> pts_1, pts_2;
    for ( int i=0;i<pts2d_1.size();i++ )
    {
        // 将像素坐标转换至相机坐标
        pts_1.push_back ( pixel2cam( pts2d_1[i], K) );
        pts_2.push_back ( pixel2cam( pts2d_2[i], K) );
//        std::cout<<"p2d: "<<pts_1.back().x<<" "<<pts_1.back().y<<std::endl;
//        std::cout<<"p2d: "<<pts_2.back().x<<" "<<pts_2.back().y<<std::endl;
//        std::cout<<std::endl;
    }

    Mat pts_4f;
//    std::cout<<"T1:\n"<<T1<<std::endl;
//    std::cout<<"T2:\n"<<T2<<std::endl;

    cv::triangulatePoints( T1, T2, pts_1, pts_2, pts_4f );

    // 转换成非齐次坐标
    for ( int i=0; i<pts_4f.cols; i++ )
    {
        Mat x = pts_4f.col(i);
        x /= x.at<float>(3,0); // 归一化
        Point3f p (
                x.at<float>(0,0),
                x.at<float>(1,0),
                x.at<float>(2,0)
        );
        points.push_back(p);
    }
}

Point2f pixel2cam ( const Point2f& p, const Mat& K )
{

    Point2f test(
                    ( p.x - K.at<float>(0,2) ) / K.at<float>(0,0),
                    ( p.y - K.at<float>(1,2) ) / K.at<float>(1,1)
            );
    //std::cout<<"K: \n"<<K<<std::endl;
    //std::cout<<"the backprojection "<<  test.x <<" "<<  test.y<<std::endl;
    return Point2f
            (
                    ( p.x - K.at<float>(0,2) ) / K.at<float>(0,0),
                    ( p.y - K.at<float>(1,2) ) / K.at<float>(1,1)
            );
}
