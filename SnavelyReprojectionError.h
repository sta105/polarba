#ifndef SnavelyReprojection_H
#define SnavelyReprojection_H

#include <iostream>
#include "ceres/ceres.h"


#include "common/tools/rotation.h"
#include "common/projection.h"

class SnavelyReprojectionError
{
public:
    SnavelyReprojectionError(double observation_x, double observation_y,double observation_angle):observed_x(observation_x),observed_y(observation_y),observed_angle(observation_angle){}

template<typename T>
    bool operator()(const T* const camera,
                const T* const point,
                T* residuals)const{                  
        // camera[0,1,2] are the angle-axis rotation
        T predictions[3];
        CamProjectionWithDistortion(camera, point, predictions);
        residuals[0] = predictions[0] - T(observed_x);
        residuals[1] = predictions[1] - T(observed_y);
        residuals[2] = sin(predictions[2] - T(observed_angle));

        return true;
    }

    static ceres::CostFunction* Create(const double observed_x, const double observed_y, const double observed_angle){
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError,3,9,6>(
            new SnavelyReprojectionError(observed_x,observed_y,observed_angle)));
    }


private:
    double observed_x;
    double observed_y;
    double observed_angle;
};

#endif // SnavelyReprojection.h

