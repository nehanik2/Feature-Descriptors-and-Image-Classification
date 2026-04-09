cmake_minimum_required(VERSION 3.5)
project( hough )
set( BASE_ROOT ${CMAKE_CURRENT_SOURCE_DIR} )
set( INC_ROOT ${BASE_ROOT}/include )
set( CSRC_ROOT ${BASE_ROOT}/csrc )
include_directories( ${INC_ROOT} )

set(CMAKE_CXX_STANDARD 17)

find_package( Eigen3 REQUIRED )
include_directories( ${EIGEN3_INCLUDE_DIR} )

enable_testing()
find_package(GTest REQUIRED)


# find CannyLibs libraries
list( APPEND CMAKE_MODULE_PATH $ENV{PA3_ROOT}/cmake/modules)
find_package( ImClsLibs REQUIRED )
include_directories( ${ImClsLibs_INCLUDE_DIR} )

add_subdirectory( ${CSRC_ROOT} )

