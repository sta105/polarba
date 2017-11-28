//
// Created by sicong on 24/11/17.
//

#ifndef CERES_CUSTOMBUNDLE_FRAME_H
#define CERES_CUSTOMBUNDLE_FRAME_H

#include <Eigen/Core>

using namespace Eigen;

class P3d{
public:
    P3d(){sharedcam.clear();};
    P3d(Vector3d _xyz, Vector3d _normal):xyz(_xyz),normal(_normal){sharedcam.clear();};
    //~P3d();
    int pindex;
    Vector3d xyz;
    Vector3d normal;
    std::vector<int> sharedcam;
};

class pixel {
public:
    pixel(){};
    pixel(int x, int y){coordinate[0] =x,coordinate[1] =y;}
    //~pixel();

    int coordinate[2]; //first two are for image coordinate
    double angle;      // azu angle
    int ptindex;
    P3d *p3d;
    double* worldpt;
    double* camerapt;
    bool triangulated = false;
};


class Frame {
public:
    Frame(){
        w = 640;
        h = 480;
        extrinsic.setIdentity();
        intrinsic.setIdentity();
        noisyextrinsic.setIdentity();
        intrinsic(0,0) = 460; //fx
        intrinsic(1,1) = 460; //fy
        intrinsic(0,2) = 320; //cx
        intrinsic(1,2) = 240; //cy
    };
    //~Frame();
    int w,h;
    Matrix4d extrinsic;
    Matrix3d intrinsic;

    Matrix4d noisyextrinsic;

    std::vector<pixel> obs;
    pixel tmppixel; // the observation to be inserted

    bool projectionvalid;
};


#endif //CERES_CUSTOMBUNDLE_FRAME_H
