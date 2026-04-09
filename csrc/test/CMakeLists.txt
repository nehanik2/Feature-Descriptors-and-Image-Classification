
enable_testing()

get_filename_component( test_SRC_ROOT ${CMAKE_CURRENT_LIST_DIR} ABSOLUTE )


add_executable( ${PROJECT_NAME}-vbow-test ${test_SRC_ROOT}/vbow_test.cc )
target_link_libraries( ${PROJECT_NAME}-vbow-test
                       ${PROJECT_NAME}-vbow                    # your library
                       ${ImClsLibs_LIBRARIES}                  # my libraries
                       Eigen3::Eigen GTest::GTest GTest::Main ) # third party dependencies
gtest_discover_tests( ${PROJECT_NAME}-vbow-test )

