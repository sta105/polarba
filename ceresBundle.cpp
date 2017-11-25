#include <iostream>
#include <fstream>
#include "ceres/ceres.h"
#include <time.h>

#include "SnavelyReprojectionError.h"
#include "common/BALProblem.h"
#include "common/BundleParams.h"


using namespace ceres;

void SetLinearSolver(ceres::Solver::Options* options, const BundleParams& params)
{
    CHECK(ceres::StringToLinearSolverType(params.linear_solver, &options->linear_solver_type));
    CHECK(ceres::StringToSparseLinearAlgebraLibraryType(params.sparse_linear_algebra_library, &options->sparse_linear_algebra_library_type));
    CHECK(ceres::StringToDenseLinearAlgebraLibraryType(params.dense_linear_algebra_library, &options->dense_linear_algebra_library_type));
    options->num_linear_solver_threads = params.num_threads;

}


void SetOrdering(BALProblem* bal_problem, ceres::Solver::Options* options, const BundleParams& params)
{
    const int num_points = bal_problem->num_points();
    const int point_block_size = bal_problem->point_block_size();
    double* points = bal_problem->mutable_points();

    const int num_cameras = bal_problem->num_cameras();
    const int camera_block_size = bal_problem->camera_block_size();
    double* cameras = bal_problem->mutable_cameras();


    if (params.ordering == "automatic")
        return;

    ceres::ParameterBlockOrdering* ordering = new ceres::ParameterBlockOrdering;

    // The points come before the cameras
    for(int i = 0; i < num_points; ++i)
       ordering->AddElementToGroup(points + point_block_size * i, 0);
       
    
    for(int i = 0; i < num_cameras; ++i)
        ordering->AddElementToGroup(cameras + camera_block_size * i, 1);

    options->linear_solver_ordering.reset(ordering);

}

void SetMinimizerOptions(Solver::Options* options, const BundleParams& params){
    options->max_num_iterations = params.num_iterations;
    options->minimizer_progress_to_stdout = true;
    options->num_threads = params.num_threads;
    // options->eta = params.eta;
    // options->max_solver_time_in_seconds = params.max_solver_time;
    
    CHECK(StringToTrustRegionStrategyType(params.trust_region_strategy,
                                        &options->trust_region_strategy_type));
    
}

void SetSolverOptionsFromFlags(BALProblem* bal_problem,
                               const BundleParams& params, Solver::Options* options){
    SetMinimizerOptions(options,params);
    SetLinearSolver(options,params);
    SetOrdering(bal_problem, options,params);
}

void BuildProblem(BALProblem* bal_problem, Problem* problem, const BundleParams& params)
{
    const int point_block_size = bal_problem->point_block_size();
    const int camera_block_size = bal_problem->camera_block_size();
    double* points = bal_problem->mutable_points();
    double* cameras = bal_problem->mutable_cameras();

    // Observations is 2 * num_observations long array observations
    // [u_1, u_2, ... u_n], where each u_i is two dimensional, the x 
    // and y position of the observation. 
    const double* observations = bal_problem->observations();

    for(int i = 0; i < bal_problem->num_observations(); ++i){
        CostFunction* cost_function;

        // Each Residual block takes a point and a camera as input 
        // and outputs a 3 dimensional Residual , the last observation is the azu angle
      
        cost_function = SnavelyReprojectionError::Create(observations[3*i + 0], observations[3*i + 1], observations[3*i + 2]);

        // If enabled use Huber's loss function. 
        LossFunction* loss_function = params.robustify ? new HuberLoss(1.0) : NULL;

        // Each observatoin corresponds to a pair of a camera and a point 
        // which are identified by camera_index()[i] and point_index()[i]
        // respectively.
        double* camera = cameras + camera_block_size * bal_problem->camera_index()[i];
        double* point = points + point_block_size * bal_problem->point_index()[i];

     
        problem->AddResidualBlock(cost_function, loss_function, camera, point);
    }

}

void SolveProblem(const char* filename, const BundleParams& params)
{
    srand(time(NULL));
    BALProblem bal_problem(filename);
    bal_problem.Datagenerator(filename);

    std::ofstream of("eigen.ply");

    of<< "ply"
      << '\n' << "format ascii 1.0"
      << '\n' << "element vertex " << bal_problem.Framebuf.size() + bal_problem.pcbuf.size()
      << '\n' << "property float x"
      << '\n' << "property float y"
      << '\n' << "property float z"
      << '\n' << "property uchar red"
      << '\n' << "property uchar green"
      << '\n' << "property uchar blue"
      << '\n' << "end_header" << std::endl;

    std::cout<<"print "<<bal_problem.pcbuf.size()<<" points"<<std::endl;
    // Export the structure (i.e. 3D Points) as white points.

    for(int i = 0; i < bal_problem.Framebuf.size(); ++i){
        Vector3d camcenter;
        //Vector3d spherecenter(0,0,3);
        camcenter = -1.0 * bal_problem.Framebuf[i].extrinsic.block<3,3>(0,0).transpose() * bal_problem.Framebuf[i].extrinsic.block<3,1>(0,3);
        //std::cout<<"sphere error: "<< (camcenter - spherecenter).norm() - 3.0 <<std::endl;
        of << camcenter(0) << ' '<< camcenter(1) <<' '<< camcenter(2) <<' ';
        of << "0 0 255\n";
    }

    for(int i = 0; i < bal_problem.pcbuf.size(); ++i){
        of << bal_problem.pcbuf[i].xyz(0) << ' '<< bal_problem.pcbuf[i].xyz(1) <<' '<< bal_problem.pcbuf[i].xyz(2) <<' ';
        of << "255 255 255\n";
    }
    of.close();


    std::ofstream of1("projection.ply");

    of1<< "ply"
      << '\n' << "format ascii 1.0"
      << '\n' << "element vertex " << bal_problem.Framebuf[0].obs.size()
      << '\n' << "property float x"
      << '\n' << "property float y"
      << '\n' << "property float z"
      << '\n' << "property uchar red"
      << '\n' << "property uchar green"
      << '\n' << "property uchar blue"
      << '\n' << "end_header" << std::endl;

    std::cout<<"print "<<bal_problem.Framebuf[0].obs.size()<<" projections"<<std::endl;
    // Export the structure (i.e. 3D Points) as white points.

    for(int i = 0; i < bal_problem.Framebuf[0].obs.size(); ++i){
        //Vector3d camcenter;
        //Vector3d spherecenter(0,0,3);
        //camcenter = -1.0 * bal_problem.Framebuf[0].extrinsic.block<3,3>(0,0).transpose() * bal_problem.Framebuf[i].extrinsic.block<3,1>(0,3);
        //std::cout<<"sphere error: "<< (camcenter - spherecenter).norm() - 3.0 <<std::endl;
        of1 << (double)bal_problem.Framebuf[0].obs[i].coordinate[0] << ' '<< (double)bal_problem.Framebuf[0].obs[i].coordinate[1] <<' '<< 0.0 <<' ';
        of1 << "0 0 255\n";
    }
    of1.close();


    bal_problem.initialization();
    //bal_problem.solveodometry();

    // show some information here ...
    std::cout << "bal problem file loaded..." << std::endl;
    std::cout << "bal problem have " << bal_problem.num_cameras() << " cameras and "
              << bal_problem.num_points() << " points. " << std::endl;
    std::cout << "Forming " << bal_problem.num_observations() << " observatoins. " << std::endl;

    // store the initial 3D cloud points and camera pose..

    std::cout << "beginning problem..." << std::endl;


    // add some noise for the intial value
    bal_problem.Normalize();

    if(!params.initial_ply.empty()){
        bal_problem.WriteToPLYFile(params.initial_ply);
    }


    bal_problem.Perturb(params.rotation_sigma, params.translation_sigma,
                        params.point_sigma,params.normal_sigma);

    std::cout << "Normalization complete..." << std::endl;

    if(!params.initial_ply.empty()){
        bal_problem.WriteToPLYFile(params.noisy_ply);
    }
    
    Problem problem;
    BuildProblem(&bal_problem, &problem, params);

    std::cout << "the problem is successfully build.." << std::endl;
   
   
    Solver::Options options;
    SetSolverOptionsFromFlags(&bal_problem, params, &options);
    options.gradient_tolerance = 1e-16;
    options.function_tolerance = 1e-16;
    Solver::Summary summary;
    Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    // write the result into a .ply file.   
    if(!params.final_ply.empty()){
        bal_problem.WriteToPLYFile(params.final_ply);  // pay attention to this: ceres doesn't copy the value into optimizer, but implement on raw data! 
    }
}

int main(int argc, char** argv)
{    
    BundleParams params(argc,argv);  // set the parameters here.
   //std::cout<<"start"<<std::endl;
    google::InitGoogleLogging(argv[0]);
    //std::cout<<"end"<<std::endl;
    std::cout << params.input << std::endl;
    if(params.input.empty()){
        std::cout << "Usage: bundle_adjuster -input <path for dataset>";
        return 1;
    }

    SolveProblem(params.input.c_str(), params);
 
    return 0;
}