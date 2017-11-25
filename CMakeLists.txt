cmake_minimum_required(VERSION 2.8)

project(ceres_customBundle)

find_package(Ceres REQUIRED)
find_package(OpenCV 3 REQUIRED)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
set(CMAKE_BUILD_TYPE "DEBUG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")


include_directories(${OpenCV_INCLUDE_DIRS}
                    ${CERES_INCLUDE_DIRS}
                    ${PROJECT_SOURCE_DIR}/common
                    ${PROJECT_SOURCE_DIR}/common/tools 
                    ${PROJECT_SOURCE_DIR}/common/flags)

add_library(BALProblem SHARED ${PROJECT_SOURCE_DIR}/common/BALProblem.cpp)
add_library(ParseCmd SHARED ${PROJECT_SOURCE_DIR}/common/flags/command_args.cpp)

#add_executable( feature_extraction feature_extraction.cpp  )
#target_link_libraries( feature_extraction ${OpenCV_LIBS} )

# add_executable( pose_estimation_2d2d pose_estimation_2d2d.cpp extra.cpp ) # use this if in OpenCV2
#add_executable( pose_estimation_2d2d pose_estimation_2d2d.cpp )
#target_link_libraries( pose_estimation_2d2d ${OpenCV_LIBS} )

# add_executable( triangulation triangulation.cpp extra.cpp) # use this if in opencv2
#add_executable( triangulation triangulation.cpp )
#target_link_libraries( triangulation ${OpenCV_LIBS} )


add_executable(${PROJECT_NAME} ceresBundle.cpp)

target_link_libraries(${PROJECT_NAME} BALProblem ParseCmd ${CERES_LIBRARIES} ${OpenCV_LIBS})
