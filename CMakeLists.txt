cmake_minimum_required(VERSION 3.0)
project(max_pool_add)

OPTION(USE_OMP "Use OpenMP" OFF)
OPTION(USE_AVX "Use AVX" OFF)

IF(USE_OMP)
    ADD_DEFINITIONS(-DUSE_OMP)
ENDIF(USE_OMP)

IF(USE_OMP)
    ADD_DEFINITIONS(-DUSE_AVX)
ENDIF(USE_OMP)


set (CMAKE_CXX_FLAGS "-fopenmp -mavx2")




add_executable(mpa main.cpp utils.cpp)

target_include_directories(mpa
    PUBLIC {PROJECT_SOURCE_DIR}
)