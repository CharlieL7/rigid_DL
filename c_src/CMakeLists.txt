cmake_minimum_required(VERSION 3.1...3.13)
if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()


project(RDL_CXX_SOURCE VERSION 1.0
	DESCRIPTION "C++ source code for rigid double layer project. Used for speeding up the numerics."
	LANGUAGES CXX)


# Set a default build type if none was specified
set(default_build_type "Release")
if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
  set(default_build_type "Debug")
endif()
 
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

ENABLE_LANGUAGE(C)

# finding libraries
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# include_directories("${PROJECT_SOURCE_DIR}/c_src")
add_library(matrix_assem STATIC
	matrix_assem.hpp
	matrix_assem.cpp
)

target_link_libraries (
	matrix_assem PUBLIC
	Eigen3::Eigen
)

target_compile_features(matrix_assem PUBLIC cxx_std_11)
