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

    bool operator()(const double* const camera,
                const double* const point,
                    double* residuals)const{
        // camera[0,1,2] are the angle-axis rotation
        double predictions[3];
        CamProjectionWithDistortion(camera, point, predictions);
        residuals[0] = predictions[0] - observed_x;
        residuals[1] = predictions[1] - observed_y;
        residuals[2] = 10.0 *sin(predictions[2] - observed_angle);

        return true;
    }

    static ceres::CostFunction* Create(const double observed_x, const double observed_y, const double observed_angle){
        return (new ceres::NumericDiffCostFunction<SnavelyReprojectionError,ceres::CENTRAL,3,9,6>(
            new SnavelyReprojectionError(observed_x,observed_y,observed_angle)));
    }


private:
    double observed_x;
    double observed_y;
    double observed_angle;
};

#endif // SnavelyReprojection.h

