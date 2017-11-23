#include "BALProblem.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>


#include "tools/random.h"
#include "tools/rotation.h"
#include "tools/random.h"
#include "projection.h"


typedef Eigen::Map<Eigen::VectorXd> VectorRef;
typedef Eigen::Map<const Eigen::VectorXd> ConstVectorRef;

template<typename T>
void FscanfOrDie(FILE *fptr, const char *format, T *value){
        int num_scanned = fscanf(fptr, format, value);
        if(num_scanned != 1)
            std::cerr<< "Invalid UW data file. ";
}

void PerturbPoint3(const double sigma, double* point)
{
  for(int i = 0; i < 3; ++i)
    point[i] += RandNormal()*sigma;
}

double Median(std::vector<double>* data){
  int n = data->size();
  std::vector<double>::iterator mid_point = data->begin() + n/2;
  std::nth_element(data->begin(),mid_point,data->end());
  return *mid_point;
}

double get_rand(double min, double max)
{
    double range = max-min;
    double p = RandDouble();
    return min+p*range;
}

BALProblem::BALProblem(const std::string& filename, bool use_quaternions){
    num_cameras_ = 2;
    num_points_ = 1000;
    num_observations_ = num_points_ * num_cameras_;

    std::cout << "Header: " << num_cameras_
            << " " << num_points_
            << " " << num_observations_;

    point3d = new double[3*num_points_];
    normal3d = new double[3*num_points_];
    point_index_ = new int[num_observations_];
    camera_index_ = new int[num_observations_];
    observations_ = new double[3 * num_observations_]; // first two for pixel position, last for polar angle


    num_parameters_ = 9 * num_cameras_ + 6 * num_points_;
    parameters_ = new double[num_parameters_];

    std::vector<Eigen::Matrix4d> posebuf;
    //set the first pose to
    // 1 0 0 0
    // 0 1 0 0
    // 0 0 1 -3
    // 0 0 0 0
    Eigen::Matrix4d firstpose;
    firstpose.setIdentity();
    firstpose.block<3,1>(0,3) << 0, 0, 3;

    //set a rand transformation
    Eigen::Matrix4d relativepose;
    Eigen::Vector3d randaxis;
    randaxis << get_rand(0,1), get_rand(0,1) , get_rand(0,1);
    randaxis.normalize();
    Eigen::AngleAxisd relativerot(get_rand(M_PI/10,M_PI/4), randaxis);
    relativepose.setIdentity();
    relativepose.block<3,3>(0,0) = relativerot.toRotationMatrix();
    relativepose(0,3) = get_rand(-0.1,0.1);
    relativepose(1,3) = get_rand(-0.1,0.1);
    relativepose(2,3) = get_rand(-0.1,0.1);


    posebuf.push_back(firstpose);

    for(int i=1;i<num_cameras_;i++){
        Eigen::Matrix4d currentpos;
        currentpos.setIdentity();
        currentpos = relativepose * firstpose;
        posebuf.push_back(currentpos);
    }

    for(int i=0;i<posebuf.size();i++){
        Eigen::Matrix4d pose = posebuf[i];
        Eigen::AngleAxisd currentrot(pose.block<3,3>(0,0));
        Eigen::Vector3d axisvec = currentrot.angle() * currentrot.axis();
        std::cout<<"Pose "<<i<<" :\n"<<pose<<std::endl;
        parameters_[i*9+0] = axisvec(0); // this is wrong, need to reset the rotaxis later
        parameters_[i*9+1] = axisvec(1);
        parameters_[i*9+2] = axisvec(2);
        parameters_[i*9+3] = pose(0,3);
        parameters_[i*9+4] = pose(1,3);
        parameters_[i*9+5] = pose(3,3);
        parameters_[i*9+6] = 1000;  //the focal length needs to be fixed later
        parameters_[i*9+7] = 0.0;  // assume there is no distortion
        parameters_[i*9+8] = 0.0;
    }

    for(int i=0;i<num_points_;i++)
    {
        double nx = get_rand(-1.0,1.0);
        double ny = get_rand(-1.0,1.0);
        double nz = get_rand(-1.0,1.0);
        double nnorm = sqrt(nx*nx + ny*ny + nz*nz);
        nx/=nnorm;
        ny/=nnorm;
        nz/=nnorm;
        //std::cout<<"nx ny nz"<<nx<<" "<<ny<<" "<<nz<<std::endl;

        double px = get_rand(-1.0,1.0);
        double py = get_rand(-1.0,1.0);
        double pz = get_rand(-1.0,1.0);

//        normal3d[i*3] = nx;
//        normal3d[i*3+1] = ny;
//        normal3d[i*3+2] = nz;
//
//        point3d[i*3] = px;
//        point3d[i*3+1] = py;
//        point3d[i*3+2] = pz;
        parameters_[num_cameras_*9 + i*6] = px;
        parameters_[num_cameras_*9 + i*6 +1] = py;
        parameters_[num_cameras_*9 + i*6 +2] = pz;
        parameters_[num_cameras_*9 + i*6 +3] = nx;
        parameters_[num_cameras_*9 + i*6 +4] = ny;
        parameters_[num_cameras_*9 + i*6 +5] = nz;

        for(int j=0;j<num_cameras_;j++)
        {
            int observations_index = i*num_cameras_+j;
            camera_index_[observations_index] = j;
            point_index_[observations_index] = i;
            projector(nx,ny,nz,px,py,pz,j,i,num_cameras_,parameters_,observations_);
        }
    }


  use_quaternions_ = use_quaternions;
}


void BALProblem::WriteToFile(const std::string& filename)const{
  FILE* fptr = fopen(filename.c_str(),"w");
  
  if(fptr == NULL)
  {
    std::cerr<<"Error: unable to open file "<< filename;
    return;
  }

  fprintf(fptr, "%d %d %d %d\n", num_cameras_, num_cameras_, num_points_, num_observations_);

  for(int i = 0; i < num_observations_; ++i){
    fprintf(fptr, "%d %d", camera_index_[i], point_index_[i]);
    for(int j = 0; j < 2; ++j){
      fprintf(fptr, " %g", observations_[2*i + j]);
    }
    fprintf(fptr,"\n");
  }

  for(int i = 0; i < num_cameras(); ++i)
  {
    double angleaxis[9];
    if(use_quaternions_){
      //OutPut in angle-axis format.
      QuaternionToAngleAxis(parameters_ + 10 * i, angleaxis);
      memcpy(angleaxis + 3, parameters_ + 10 * i + 4, 6 * sizeof(double));
    }else{
      memcpy(angleaxis, parameters_ + 9 * i, 9 * sizeof(double));
    }
    for(int j = 0; j < 9; ++j)
    {
      fprintf(fptr, "%.16g\n",angleaxis[j]);
    }
  }

  const double* points = parameters_ + camera_block_size() * num_cameras_;
  for(int i = 0; i < num_points(); ++i){
    const double* point = points + i * point_block_size();
    for(int j = 0; j < point_block_size(); ++j){
      fprintf(fptr,"%.16g\n",point[j]);
    }
  }

  fclose(fptr);
}

// Write the problem to a PLY file for inspection in Meshlab or CloudCompare
void BALProblem::WriteToPLYFile(const std::string& filename)const{
  std::ofstream of(filename.c_str());

  of<< "ply"
    << '\n' << "format ascii 1.0"
    << '\n' << "element vertex " << num_cameras_ + num_points_
    << '\n' << "property float x"
    << '\n' << "property float y"
    << '\n' << "property float z"
    << '\n' << "property uchar red"
    << '\n' << "property uchar green"
    << '\n' << "property uchar blue"
    << '\n' << "end_header" << std::endl;

    // Export extrinsic data (i.e. camera centers) as green points.
    double angle_axis[3];
    double center[3];
    for(int i = 0; i < num_cameras(); ++i){
      const double* camera = cameras() + camera_block_size() * i;
      CameraToAngelAxisAndCenter(camera, angle_axis, center);
      of << center[0] << ' ' << center[1] << ' ' << center[2] <<" "<< "0 255 0" << '\n';
    }

    // Export the structure (i.e. 3D Points) as white points.
    const double* points = parameters_ + camera_block_size() * num_cameras_;
    for(int i = 0; i < num_points(); ++i){
      const double* point = points + i * point_block_size();
      for(int j = 0; j < point_block_size(); ++j){
        if(j<3)
        {of << point[j] << ' ';}
      }
      of << "255 255 255\n";
    }
    of.close();
}

void BALProblem::CameraToAngelAxisAndCenter(const double* camera, 
                                            double* angle_axis,
                                            double* center) const{
    VectorRef angle_axis_ref(angle_axis,3);
    if(use_quaternions_){
      QuaternionToAngleAxis(camera, angle_axis);
    }else{
      angle_axis_ref = ConstVectorRef(camera,3);
    }

    // c = -R't
    Eigen::VectorXd inverse_rotation = -angle_axis_ref;
    std::cout<<"rotation:\n"<<inverse_rotation<<std::endl;
    AngleAxisRotatePoint(inverse_rotation.data(),
                         camera + camera_block_size() - 6,
                         center);
    std::cout<<"The camera center:"<<center[0]<<" "<<center[1]<<" "<<center[2]<<std::endl;
    VectorRef(center,3) *= -1.0;
    std::cout<<"The camera center:"<<center[0]<<" "<<center[1]<<" "<<center[2]<<std::endl;
}

void BALProblem::AngleAxisAndCenterToCamera(const double* angle_axis,
                                            const double* center,
                                            double* camera) const{
    ConstVectorRef angle_axis_ref(angle_axis,3);
    if(use_quaternions_){
      AngleAxisToQuaternion(angle_axis,camera);
    }
    else{
      VectorRef(camera, 3) = angle_axis_ref;
    }

    // t = -R * c 
    AngleAxisRotatePoint(angle_axis,center,camera+camera_block_size() - 6);
    VectorRef(camera + camera_block_size() - 6,3) *= -1.0;
}

void BALProblem::Normalize(){
  // Compute the marginal median of the geometry
  std::vector<double> tmp(num_points_);
  Eigen::Vector3d median;
  double* points = mutable_points();
  for(int i = 0; i < 3; ++i){
    for(int j = 0; j < num_points_; ++j){
      tmp[j] = points[3 * j + i];      
    }
    median(i) = Median(&tmp);
  }

  for(int i = 0; i < num_points_; ++i){
    VectorRef point(points + 3 * i, 3);
    tmp[i] = (point - median).lpNorm<1>();
  }

  const double median_absolute_deviation = Median(&tmp);

  // Scale so that the median absolute deviation of the resulting
  // reconstruction is 100

  const double scale = 100.0 / median_absolute_deviation;

  // X = scale * (X - median)
  for(int i = 0; i < num_points_; ++i){
    VectorRef point(points + 3 * i, 3);
    point = scale * (point - median);
  }

  double* cameras = mutable_cameras();
  double angle_axis[3];
  double center[3];
  for(int i = 0; i < num_cameras_ ; ++i){
    double* camera = cameras + camera_block_size() * i;
    CameraToAngelAxisAndCenter(camera, angle_axis, center);
    // center = scale * (center - median)
    VectorRef(center,3) = scale * (VectorRef(center,3)-median);
    AngleAxisAndCenterToCamera(angle_axis, center,camera);
  }
}

void BALProblem::Perturb(const double rotation_sigma, 
                         const double translation_sigma,
                         const double point_sigma,
                        const double normal_sigma){
   assert(point_sigma >= 0.0);
   assert(rotation_sigma >= 0.0);
   assert(translation_sigma >= 0.0);
   assert(normal_sigma >= 0.0);

   double* points = mutable_points();

    if(normal_sigma > 0){
        for(int i = 0; i < num_points_; ++i){
            PerturbPoint3(point_sigma, points + 6 * i );
        }
    }

    if(point_sigma > 0){
     for(int i = 0; i < num_points_; ++i){
       PerturbPoint3(point_sigma, points + 6 * i + 3 );
     }
   }

   for(int i = 0; i < num_cameras_; ++i){
     double* camera = mutable_cameras() + camera_block_size() * i;

     double angle_axis[3];
     double center[3];
     // Perturb in the rotation of the camera in the angle-axis
     // representation
     CameraToAngelAxisAndCenter(camera, angle_axis, center);
     if(rotation_sigma > 0.0){
       PerturbPoint3(rotation_sigma, angle_axis);
     }
     AngleAxisAndCenterToCamera(angle_axis, center, camera);

     if(translation_sigma > 0.0)
        PerturbPoint3(translation_sigma, camera + camera_block_size() - 6);
   }
}