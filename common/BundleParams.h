#ifndef BUNDLEPARAMS_H
#define BUNDLEPARAMS_H

#include <string>
#include "flags/command_args.h"

using namespace std;

struct BundleParams{
public:
    BundleParams(int argc, char** argv);
    virtual ~BundleParams(){};

public:
    string input;
    string trust_region_strategy;
    string linear_solver;
    string sparse_linear_algebra_library;
    string dense_linear_algebra_library;


    string ordering; // marginalization ..

    bool robustify; // loss function
    // double eta;
    int num_threads;  // default = 1
    int num_iterations;
    
    // for making noise
    int random_seed;
    double rotation_sigma;
    double translation_sigma;
    double point_sigma;
    double normal_sigma;

    // for point cloud file...
    string initial_ply;
    string final_ply;
    string noisy_ply;

    CommandArgs arg;

};

 BundleParams::BundleParams(int argc, char** argv)
 {  
    arg.param("input", input, "", "file which will be processed");
    arg.param("trust_region_strategy", trust_region_strategy, "dogleg",
              "Options are: levenberg_marquardt, dogleg.");
    arg.param("linear_solver", linear_solver, "dense_schur",                             // iterative schur and cgnr(pcg) leave behind...
              "Options are: sparse_schur, dense_schur, sparse_normal_cholesky");
    
    arg.param("sparse_linear_algebra_library", sparse_linear_algebra_library, "suite_sparse", "Options are: suite_sparse and cx_sparse.");
    arg.param("dense_linear_algebra_library", dense_linear_algebra_library, "eigen", "Options are: eigen and lapack.");
    
    
    arg.param("ordering",ordering,"user","Options are: automatic, user.");
    arg.param("robustify", robustify, true, "Use a robust loss function");
    

    arg.param("num_threads",num_threads,1, "Number of threads.");
    arg.param("num_iterations", num_iterations,1000, "Number of iterations.");

    arg.param("rotation_sigma", rotation_sigma, 0.2, "Standard deviation of camera rotation "
              "perturbation.");
    arg.param("translation_sigma", translation_sigma,5, "translation perturbation.");
    arg.param("point_sigma",point_sigma,6,"Standard deviation of the point "
              "perturbation.");
    arg.param("normal_sigma",normal_sigma,0.2,"Standard deviation of the normal "
             "perturbation.");

    arg.param("random_seed", random_seed, 38401,"Random seed used to set the state ");
    arg.param("initial_ply", initial_ply,"initial.ply","Export the BAL file data as a PLY file.");
     arg.param("noisy_ply", noisy_ply, "noisy.ply", "Export the noisy BAL file data as a PLY");
    arg.param("final_ply", final_ply, "final.ply", "Export the refined BAL file data as a PLY");


    arg.parseArgs(argc, argv);
 }

#endif