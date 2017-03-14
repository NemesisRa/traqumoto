cmake_minimum_required(VERSION 3.6)
project(traqumoto)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)

set(SOURCE_FILES main.cpp)
add_executable(traqumoto ${SOURCE_FILES})

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(traqumoto ${OpenCV_LIBS})


