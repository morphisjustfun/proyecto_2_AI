cmake_minimum_required(VERSION 3.23)
project(proyecto_2)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++ -static-libgcc")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -gdwarf-4")

add_executable(proyecto_2 main.cpp utils/cfunctions.cpp utils/cfunctions.h classes/dataframe.cpp classes/dataframe.h classes/ds_tree.cpp classes/ds_tree.h classes/kdtree.cpp classes/kdtree.h)