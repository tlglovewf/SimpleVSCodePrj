cmake_minimum_required(VERSION 2.8)
project(CV_Project)

#set cmake build type
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Debug)
ENDIF()

#print build type
MESSAGE("Build  type: " ${CMAKE_BUILD_TYPE})

#set debug compile setting
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O0 -g  -march=native ")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O0 -g -march=native")

#release compile setting
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3  -march=native ")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

#Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

#set cmake modules find path
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

#ex find opencv 
find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)

if(NOT Pangolin_FOUND)
   message(FATAL_ERROR "Pangolin not found.")
else()
   message("pangolin path : " ${Pangolin_INCLUDE_DIRS})
endif()

#set include file search path
include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
)

#set lib link path 
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

#build library 

set(MYLIB "mylibs")

add_library(${MYLIB} SHARED
src/test.cpp
)

#link libs
target_link_libraries(${MYLIB}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
#${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
)

#set program running path 
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/OutPut)

#build exe 
add_executable(te
main.cpp)
target_link_libraries(te ${MYLIB})