#ifndef PROJECTION_H
#define PROJECTION_H

#include "tools/rotation.h"
#include <Eigen/Core>
#include <Eigen/Geometry>

// camera : 9 dims array with 
// [0-2] : angle-axis rotation 
// [3-5] : translateion
// [6-8] : camera parameter, [6] focal length, [7-8] second and forth order radial distortion
// point : 3D location. 
// predictions : 2D predictions with center of the image plane.


bool projector(double nx,double ny,double nz,double px,double py,double pz, int camindex, int pointindex, int camnum, double* parameters_,double* observations)
{
    // Rodrigues' formula
    //std::cout<<"nx ny nz px py pz:"<<nx<<" "<<ny<<" "<<nz<<" "<<px<<" "<<py<<" "<<pz<<std::endl;
    double p[3];
    double point[3];
    double camera[9];
    point[0] = px;
    point[1] = py;
    point[2] = pz;
    for(int i=0;i<9;i++){
        camera[i] = parameters_[camindex*9+i];
    }
    AngleAxisRotatePoint(camera, point, p);
    // camera[3,4,5] are the translation
    p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];

    // Compute the center fo distortion
    double xp = -p[0]/p[2];
    double yp = -p[1]/p[2];

    // Apply second and fourth order radial distortion
    const double& l1 = camera[7];
    const double& l2 = camera[8];

    double r2 = xp*xp + yp*yp;
    double distortion = 1.0 + r2 * (l1 + l2 * r2);

    const double& focal = camera[6];
    observations[pointindex*camnum*3 + camindex*3] = focal * distortion * xp;
    observations[pointindex*camnum*3 + camindex*3 +1] = focal * distortion * yp;


    // for computing the polar angle
    Eigen::Vector3d normal(nx, ny, nz);
    Eigen::Vector3d rotvec(parameters_[camindex*9], parameters_[camindex*9+1], parameters_[camindex*9+2]);
    double angle = rotvec.norm();
    if(abs(angle)<1e-3) rotvec<<1.0,0.0,0.0;
    rotvec.normalize();
    Eigen::AngleAxisd camrotaxis(angle, rotvec);
    //std::cout<<"the rotation: "<<camrotaxis.matrix()<< std::endl;
    Eigen::Vector3d normalcam = camrotaxis * normal;
    //std::cout<<"normal projection: "<<normalcam.head(2).transpose()<<std::endl;
    observations[pointindex*camnum*3 + camindex*3 +2] = acos(normalcam(1)/normalcam.head(2).norm());
    //std::cout<<"The pixel:"<< observations[pointindex*camnum*3 + camindex*3]<<" "<< observations[pointindex*camnum*3 + camindex*3 +1]<<" "<<observations[pointindex*camnum*3 + camindex*3 +2]<<std::endl;
    return true;
}

inline bool CamProjectionWithDistortion(const double* camera, const double* point, double* predictions){
    // Rodrigues' formula
    double p[3];
    AngleAxisRotatePoint(camera, point, p);
    // camera[3,4,5] are the translation
    p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];

    // Compute the center fo distortion
    double xp = -p[0]/p[2];
    double yp = -p[1]/p[2];

    // Apply second and fourth order radial distortion
    const double& l1 = camera[7];
    const double& l2 = camera[8];

    double r2 = xp*xp + yp*yp;
    double distortion = double(1.0) + r2 * (l1 + l2 * r2);

    const double& focal = camera[6];
    predictions[0] = focal * distortion * xp;
    predictions[1] = focal * distortion * yp;


    Eigen::Vector3d normal(point[3], point[4], point[5]);
    Eigen::Vector3d rotvec(camera[0], camera[1], camera[2]);
    double angle = rotvec.norm();
    if(abs(angle)<1e-3) rotvec<<1.0,0.0,0.0;
    rotvec.normalize();
    Eigen::AngleAxisd camrotaxis(angle, rotvec);
    //std::cout<<"the rotation: "<<camrotaxis.matrix()<< std::endl;
    Eigen::Vector3d normalcam = camrotaxis * normal;
    //std::cout<<"normal projection: "<<normalcam.head(2).transpose()<<std::endl;
    predictions[2] = acos(normalcam(1)/normalcam.head(2).norm());

    return true;
}



#endif // projection.h