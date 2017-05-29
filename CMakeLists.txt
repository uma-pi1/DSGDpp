cmake_minimum_required(VERSION 2.6)

project(MF C CXX)

# add a NativeRelease build type
SET( CMAKE_CXX_FLAGS_NATIVERELEASE "-O3 -DNDEBUG -march=native -ftree-vectorize" CACHE STRING
    "Flags used by the C++ compiler during native builds."
    FORCE )
SET( CMAKE_C_FLAGS_NATIVERELEASE "-O3 -DNDEBUG -march=native -ftree-vectorize" CACHE STRING
    "Flags used by the C compiler during native builds."
    FORCE )
SET( CMAKE_EXE_LINKER_FLAGS_NATIVERELEASE "" CACHE STRING
    "Flags used for linking binaries during native builds."
    FORCE )
SET( CMAKE_SHARED_LINKER_FLAGS_NATIVERELEASE "" CACHE STRING
    "Flags used by the shared libraries linker during native builds."
    FORCE )
MARK_AS_ADVANCED(
    CMAKE_CXX_FLAGS_NATIVERELEASE
    CMAKE_C_FLAGS_NATIVERELEASE
    CMAKE_EXE_LINKER_FLAGS_NATIVERELEASE
    CMAKE_SHARED_LINKER_FLAGS_NATIVERELEASE )

if(CMAKE_CONFIGURATION_TYPES)
  list(APPEND CMAKE_CONFIGURATION_TYPES NativeRelease)
  list(REMOVE_DUPLICATES CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_CONFIGURATION_TYPES "${CMAKE_CONFIGURATION_TYPES}" CACHE STRING
     "Add the native configuration"
     FORCE)
endif()

## default build type is native
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE NativeRelease CACHE STRING
      "Choose the type of build. Options are: None Debug NativeRelease Release RelWithDebInfo MinSizeRel."
      FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)
MESSAGE(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

# handy stuff for eclipse
IF(CMAKE_COMPILER_IS_GNUCC)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fmessage-length=0")
ENDIF(CMAKE_COMPILER_IS_GNUCC)
IF(CMAKE_COMPILER_IS_GNUCXX)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmessage-length=0")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)

# definitions for visual studio
add_definitions(-D_CRT_SECURE_NO_WARNINGS)

# my own includes/libs
#include_directories(~/local/include)############
#link_directories(~/local/lib)####################3
SET(RG_LIBRARIES rg)
SET(MPI2_LIBRARIES mpi2)

# MPI
FIND_PACKAGE(MPI REQUIRED)
SET(CMAKE_C_COMPILER ccache)
SET(CMAKE_C_COMPILER_ARG1 mpicc)
SET(CMAKE_CXX_COMPILER ccache)
SET(CMAKE_CXX_COMPILER_ARG1 mpicxx)

# search for boost library
find_package(Boost 1.63 COMPONENTS serialization mpi thread program_options regex filesystem chrono)	
IF(NOT ${Boost_FOUND})
  SET(BOOST_ROOT ~/local) # default
  SET(Boost_NO_SYSTEM_PATHS ON) # force to use own build
  find_package(Boost 1.63 COMPONENTS serialization mpi thread program_options regex filesystem chrono REQUIRED)	
ENDIF(NOT ${Boost_FOUND})
link_directories(${Boost_LIBRARY_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

# search for LAPACK
set(BLAS_VENDOR ATLAS)
find_package(BLAS REQUIRED)
FIND_PACKAGE(LAPACK REQUIRED)
set(lapack_LIBRARIES /usr/lib/atlas-base/liblapack_atlas.so)

# This project uses Apache v2 licence, so we cannot include GSL.
# If nevertheless you want to use GSL for ALS (works better):
# 1. uncomment the following lines
# 2. add a FindGSL.cmake file in the projects root folder
# 3. uncomment in als_impl.cc the #include <gsl/gsl_linalg.h> and the #define ALS_USE_GSL 
### search for GSL
# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${MF_SOURCE_DIR})
# find_package(GSL REQUIRED)
# include_directories(${GSL_INCLUDE_DIRS} ${GSLCBLAS_INCLUDE_DIRS})

# include directories
include_directories(${MF_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${MF_SOURCE_DIR}/extern/include)

# install directories
SET(LIB_INSTALL_DIR lib)
SET(INCLUDE_INSTALL_DIR include)

# go
configure_file("${MF_SOURCE_DIR}/log4j.properties" "${CMAKE_CURRENT_BINARY_DIR}/")

add_subdirectory(mf)
add_subdirectory(tools)
add_subdirectory(examples)


