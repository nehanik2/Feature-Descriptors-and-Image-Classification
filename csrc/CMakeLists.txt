
get_filename_component( vbow_SRC_ROOT ${CMAKE_CURRENT_LIST_DIR} ABSOLUTE )


set( vbow_HEADERS
    ${INC_ROOT}/vbow/vbow.h
)

set( vbow_SRCS
    ${vbow_SRC_ROOT}/vbow.cc
)

add_library( ${PROJECT_NAME}-vbow SHARED ${vbow_HEADERS} ${vbow_SRCS} )
target_link_libraries( ${PROJECT_NAME}-vbow ${ImClsLibs_LIBRARIES} Eigen3::Eigen )

# tests
add_subdirectory( ${vbow_SRC_ROOT}/test )

