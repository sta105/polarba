#include "BALProblem.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>

#include "tools/random.h"
#include "tools/rotation.h"
#include "tools/random.h"
#include "projection.h"
#include "tools/triangulation.h"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>


typedef Eigen::Map<Eigen::VectorXd> VectorRef;
typedef Eigen::Map<const Eigen::VectorXd> ConstVectorRef;

using namespace Eigen;
using namespace cv;


cv::Mat toCvMat(const Eigen::Matrix3d &m)
{
    cv::Mat cvMat(3,3,CV_32F);
    for(int i=0;i<3;i++)
        for(int j=0; j<3; j++)
            cvMat.at<float>(i,j)=m(i,j);

    return cvMat.clone();
}

cv::Mat toCvMat(const Eigen::Matrix<double,4,4> &m)
{
    cv::Mat cvMat(4,4,CV_32F);
    for(int i=0;i<4;i++)
        for(int j=0; j<4; j++)
            cvMat.at<float>(i,j)=m(i,j);

    return cvMat.clone();
}

Eigen::Matrix<double,3,1> toVector3d(const cv::Mat &cvVector)
{
    Eigen::Matrix<double,3,1> v;
    v << cvVector.at<float>(0), cvVector.at<float>(1), cvVector.at<float>(2);

    return v;
}

Eigen::Matrix<double,3,1> toVector3d(const cv::Point3f &cvPoint)
{
    Eigen::Matrix<double,3,1> v;
    v << cvPoint.x, cvPoint.y, cvPoint.z;

    return v;
}

Eigen::Matrix<double,3,3> toMatrix3d(const cv::Mat &cvMat3)
{
    Eigen::Matrix<double,3,3> M;

    M << cvMat3.at<float>(0,0), cvMat3.at<float>(0,1), cvMat3.at<float>(0,2),
            cvMat3.at<float>(1,0), cvMat3.at<float>(1,1), cvMat3.at<float>(1,2),
            cvMat3.at<float>(2,0), cvMat3.at<float>(2,1), cvMat3.at<float>(2,2);

    return M;
}

Eigen::Matrix<double,3,3> MatdtoMatrix3d(const cv::Mat &cvMat3)
{
    Eigen::Matrix<double,3,3> M;

    M << cvMat3.at<double>(0,0), cvMat3.at<double>(0,1), cvMat3.at<double>(0,2),
            cvMat3.at<double>(1,0), cvMat3.at<double>(1,1), cvMat3.at<double>(1,2),
            cvMat3.at<double>(2,0), cvMat3.at<double>(2,1), cvMat3.at<double>(2,2);

    return M;
}

Eigen::Matrix<double,3,1> MatdtoVector3d(const cv::Mat &cvVector)
{
    Eigen::Matrix<double,3,1> v;
    v << cvVector.at<double>(0), cvVector.at<double>(1), cvVector.at<double>(2);

    return v;
}

void FscanfOrDie(FILE *fptr, const char *format, double *value){
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

void cameragenerator(const Vector3d center, const double rad, const Vector3d lastcam, double disparity, Vector3d& result){
    Vector2d lasttheta,currenttheta;
    Vector3d lastvec;
    lastvec = lastcam - center;
    lasttheta(0) = acos(lastvec(2)/lastvec.norm());
    if(std::isnan(lastvec(1)/lastvec(0))) lasttheta(1)= 0.0;
    else lasttheta(1) = atan(lastvec(1)/lastvec(0));
    //std::cout<<"lasttheta:"<<lasttheta.transpose()<<std::endl;
    currenttheta(0) = lasttheta(0) + get_rand(disparity/2,disparity);//(RandDouble() * disparity/2) + disparity/2;
    currenttheta(1) = lasttheta(1) + get_rand(disparity/2,disparity);//(RandDouble() * disparity/2) + disparity/2;
    //std::cout<<"currenttheta:"<<currenttheta.transpose()<<std::endl;
    result(0) = rad * sin(currenttheta(0)) * cos(currenttheta(1));
    result(1) = rad * sin(currenttheta(0)) * sin(currenttheta(1));
    result(2) = rad * cos(currenttheta(0));
    result = result + center;

    std::cout<<" sphere error:"<<(result - center).norm() - rad<<std::endl;

    assert(abs((result - center).norm() - rad)<1e-5);
}

void posegenerator(const Matrix4d lastpose, const Vector3d center, const double rad, Matrix4d& currentpose){

    currentpose.setIdentity();
    Vector3d lastcam = -1.0 * lastpose.block<3,3>(0,0).transpose() * lastpose.block<3,1>(0,3);
    Vector3d currentcam;
    double disparity = 20.0/ 180.0 * M_PIl;
    cameragenerator(center,rad,lastcam,disparity,currentcam);

    Vector3d wzaxis(0,0,1);
    Vector3d zaxis = center - currentcam;
    zaxis.normalize();
    Vector3d yaxis = wzaxis.cross(zaxis);
    Vector3d xaxis = yaxis.cross(zaxis);

    xaxis.normalize();
    yaxis.normalize();
    zaxis.normalize();

    currentpose.block<3,1>(0,0) = xaxis;
    currentpose.block<3,1>(0,1) = yaxis;
    currentpose.block<3,1>(0,2) = zaxis;
    currentpose.block<3,1>(0,3) = -1.0 * currentpose.block<3,3>(0,0) * currentcam;
}

void BALProblem::Datagenerator(const std::string& filename)
{
    num_cameras_ = 4;
    num_points_ = 500;
    num_observations_ = num_points_ * num_cameras_;

    // randomizing some camera poses;
    Frame firstframe;
    Framebuf.push_back(firstframe);
    Vector3d spherecenter(0,0,5.0);
    for(int i=0;i<num_cameras_-1;i++){
        Frame polarframe;
        Matrix4d currentpose;
        posegenerator(Framebuf.back().extrinsic, spherecenter , 5.0 , currentpose);
        polarframe.extrinsic = currentpose;
        Framebuf.push_back(polarframe);
    }

    for(auto polarframe:Framebuf){
        std::cout<<"frame pose:\n"<<polarframe.extrinsic<<std::endl;
    }

    int validpt = 0;
    while(validpt<num_points_)
    {
        Vector3d wp;
        Vector3d wn;

        wn(0) = get_rand(-1.0,1.0);
        wn(1) = get_rand(-1.0,1.0);
        wn(2) = get_rand(-1.0,1.0);
        wn.normalize();
        //std::cout<<"nx ny nz"<<nx<<" "<<ny<<" "<<nz<<std::endl;

        wp(0) = get_rand(-1.0,1.0);
        wp(1) = get_rand(-2.0,2.0);
        wp(2) = get_rand(-1.0,1.0);
        wp += spherecenter;

        //bool projector(double nx,double ny,double nz,double px,double py,double pz, int camindex, int pointindex, int camnum, double* parameters_,double* observations)
        int validobs = 0;
        for(int i=0;i<num_cameras_;i++){
            if(eigenprojector(wn, wp, validpt , Framebuf[i])) validobs++;
        }
        if (validobs>=2)
        {
            P3d pt3(wp,wn);
            for(int i=0;i<num_cameras_;i++)
            {
                if(Framebuf[i].projectionvalid)
                {
                    pt3.sharedcam.push_back(i);
                    Framebuf[i].obs.push_back(Framebuf[i].tmppixel);
                }
            }
            pt3.pindex = validpt;
            gtpcbuf.push_back(pt3);
            validpt++;
        }
    }
}



//BALProblem::BALProblem(const std::string& filename, bool use_quaternions){
////    num_cameras_ = 4;
////    num_points_ = 500;
////    num_observations_ = num_points_ * num_cameras_;
////
////    std::cout << "Header: " << num_cameras_
////            << " " << num_points_
////            << " " << num_observations_;
////
////    point3d = new double[3*num_points_];
////    normal3d = new double[3*num_points_];
////    point_index_ = new int[num_observations_];
////    camera_index_ = new int[num_observations_];
////    observations_ = new double[3 * num_observations_]; // first two for pixel position, last for polar angle
////
////
////    num_parameters_ = 9 * num_cameras_ + 6 * num_points_;
////    parameters_ = new double[num_parameters_];
////
////    std::vector<Eigen::Matrix4d> posebuf;
////    //set the first pose to
////    // 1 0 0 0
////    // 0 1 0 0
////    // 0 0 1 3
////    // 0 0 0 0
////    // then the camera center of the first camera will be 0 0 -3;
////    Eigen::Matrix4d firstpose;
////    firstpose.setIdentity();
////    firstpose.block<3,1>(0,3) << 0, 0, 3;
////
////    //set a rand relative transformation
////    Eigen::Matrix4d relativepose;
////    Eigen::Vector3d randaxis;
////    randaxis << get_rand(0,1), get_rand(0,1) , get_rand(0,1);
////    randaxis.normalize();
////    Eigen::AngleAxisd relativerot(get_rand(M_PI/10,M_PI/4), randaxis);
////    relativepose.setIdentity();
////    relativepose.block<3,3>(0,0) = relativerot.toRotationMatrix();
////    relativepose(0,3) = get_rand(-0.1,0.1);
////    relativepose(1,3) = get_rand(-0.1,0.1);
////    relativepose(2,3) = get_rand(-0.1,0.1);
////
////
////    posebuf.push_back(firstpose);
////
////    for(int i=1;i<num_cameras_;i++){
////        Eigen::Matrix4d currentpos;
////        currentpos.setIdentity();
////        currentpos = relativepose * posebuf.back();
////        posebuf.push_back(currentpos);
////    }
////
////    for(int i=0;i<posebuf.size();i++){
////        Eigen::Matrix4d pose = posebuf[i];
////        Eigen::AngleAxisd currentrot(pose.block<3,3>(0,0));
////        Eigen::Vector3d axisvec = currentrot.angle() * currentrot.axis();
////        //std::cout<<"\nPose "<<i<<" :\n"<<pose<<std::endl;
////        parameters_[i*9+0] = axisvec(0); // this is wrong, need to reset the rotaxis later
////        parameters_[i*9+1] = axisvec(1);
////        parameters_[i*9+2] = axisvec(2);
////        parameters_[i*9+3] = pose(0,3);
////        parameters_[i*9+4] = pose(1,3);
////        parameters_[i*9+5] = pose(2,3);
////        parameters_[i*9+6] = 600;  //the focal length needs to be fixed later
////        parameters_[i*9+7] = 0.0;  // assume there is no distortion
////        parameters_[i*9+8] = 0.0;
////    }
////
////    for(int i=0;i<num_points_;i++)
////    {
////        double nx = get_rand(-1.0,1.0);
////        double ny = get_rand(-1.0,1.0);
////        double nz = get_rand(-1.0,1.0);
////        double nnorm = sqrt(nx*nx + ny*ny + nz*nz);
////        nx/=nnorm;
////        ny/=nnorm;
////        nz/=nnorm;
////        //std::cout<<"nx ny nz"<<nx<<" "<<ny<<" "<<nz<<std::endl;
////
////        double px = get_rand(-1.0,1.0);
////        double py = get_rand(-1.0,1.0);
////        double pz = get_rand(-1.0,1.0);
////
//////        normal3d[i*3] = nx;
//////        normal3d[i*3+1] = ny;
//////        normal3d[i*3+2] = nz;
//////
//////        point3d[i*3] = px;
//////        point3d[i*3+1] = py;
//////        point3d[i*3+2] = pz;
////        parameters_[num_cameras_*9 + i*6] = px;
////        parameters_[num_cameras_*9 + i*6 +1] = py;
////        parameters_[num_cameras_*9 + i*6 +2] = pz;
////        parameters_[num_cameras_*9 + i*6 +3] = nx;
////        parameters_[num_cameras_*9 + i*6 +4] = ny;
////        parameters_[num_cameras_*9 + i*6 +5] = nz;
////
////        for(int j=0;j<num_cameras_;j++)
////        {
////            int observations_index = i*num_cameras_+j;
////            camera_index_[observations_index] = j;
////            point_index_[observations_index] = i;
////            projector(nx,ny,nz,px,py,pz,j,i,num_cameras_,parameters_,observations_);
////        }
////    }
////
////
//  use_quaternions_ = use_quaternions;
//}


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
    std::cout<<"translation:"<<camera[3]<<" "<<camera[4]<<" "<<camera[5]<<std::endl;
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
      tmp[j] = points[6 * j + i];
    }
    median(i) = Median(&tmp);
  }

  for(int i = 0; i < num_points_; ++i){
    VectorRef point(points + 6 * i, 3);
    tmp[i] = (point - median).lpNorm<1>();
  }

  const double median_absolute_deviation = Median(&tmp);

  // Scale so that the median absolute deviation of the resulting
  // reconstruction is 100

  const double scale = 100.0 / median_absolute_deviation;

  // X = scale * (X - median)
  for(int i = 0; i < num_points_; ++i){
    VectorRef point(points + 6 * i, 3);
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

void BALProblem::Perturbsfm(const double rotation_sigma,
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

bool BALProblem::initialization()
{
    Mat R,t,mask;
    Matrix3d intrinsic = Framebuf[0].intrinsic;
    Mat K = toCvMat(intrinsic);
    std::vector<Point2f> points1,points2;
    points1.clear();
    points2.clear();
    findmatches(0,1,points1,points2);
    std::cout<<"how many matches found:"<<points1.size()<<std::endl;
    Point2d principal_point ( intrinsic(0,2), intrinsic(1,2) );
    double focal_length = intrinsic(0,0);

    Mat essential_matrix = findEssentialMat( points1, points2, focal_length , principal_point, RANSAC );
    std::cout<<"essential_matrix:\n"<<essential_matrix<<std::endl;


    recoverPose (essential_matrix, points1, points2, R, t, focal_length, principal_point );

    std::cout<<"initialization result:\n"<<R<<std::endl;
    std::cout<<"translation:\n"<<t<<std::endl;

    std::vector<Point3f> triangulatedPoints;
    triangulatedPoints.clear();
    //recoverPose( essential_matrix, points1, points2, K, R, t, 1000, mask , triangulatedPoints);
    Framebuf[1].noisyextrinsic.block<3,3>(0,0) = MatdtoMatrix3d(R);
    Framebuf[1].noisyextrinsic.block<3,1>(0,3) = MatdtoVector3d(t);

    triangulation(points1, points2, R, t, K, triangulatedPoints);

    for(int i=0;i<triangulatedPoints.size();i++)
    {
        P3d addpt;
        addpt.xyz    = toVector3d(triangulatedPoints.[i]);
        addpt.normal = Vector3d(0,0,1); // to be added;
        addpt.pindex = 0;//to be added;
        pcbuf.push_back(addpt);
    }


    std::ofstream of2("intial.ply");

    of2<< "ply"
       << '\n' << "format ascii 1.0"
       << '\n' << "element vertex " << triangulatedPoints.size()+2
       << '\n' << "property float x"
       << '\n' << "property float y"
       << '\n' << "property float z"
       << '\n' << "property uchar red"
       << '\n' << "property uchar green"
       << '\n' << "property uchar blue"
       << '\n' << "end_header" << std::endl;

    std::cout<<"print "<<triangulatedPoints.size()<<" points"<<std::endl;
    // Export the structure (i.e. 3D Points) as white points.
    of2 << 0 << ' '<< 0 <<' '<< 0 <<' ';
    of2 << "255 255 255\n";

    Matrix3d cam1rot = Framebuf[1].noisyextrinsic.block<3,3>(0,0);
    Vector3d cam1trans = Framebuf[1].noisyextrinsic.block<3,1>(0,3);

    Vector3d cam1center = -1.0 * cam1rot.transpose() * cam1trans;
    //std::cout<<"the first cam center is :"<<cam1center.transpose()<<std::endl;

    of2 << cam1center(0) << ' '<< cam1center(1) <<' '<< cam1center(2) <<' ';
        of2 << "255 255 255\n";
    for(int i = 0; i < triangulatedPoints.size(); ++i){
        of2 << triangulatedPoints[i].x << ' '<< triangulatedPoints[i].y <<' '<< triangulatedPoints[i].z <<' ';
        of2 << "0 0 255\n";
    }
    of2.close();

    return true;
}

bool BALProblem::findmatches(const int indexi,const int indexj,std::vector<Point2f>& points1,std::vector<Point2f>& points2)
{
    if(indexi>=Framebuf.size()&&indexj>=Framebuf.size())return false;

    for(auto obs1 : Framebuf[indexi].obs)
    {
        for(auto obs2 : Framebuf[indexj].obs)
        {
            if(obs1.ptindex == obs2.ptindex)
            {
                Point2f point1(obs1.coordinate[0],obs1.coordinate[1]);
                Point2f point2(obs2.coordinate[0],obs2.coordinate[1]);
                points1.push_back(point1);
                points2.push_back(point2);
                break;
            }

        }
    }
    return true;
}


bool BALProblem::epnp()
{

    return true;
}