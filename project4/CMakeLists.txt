cmake_minimum_required(VERSION 3.10)

project(SoftmaxClassifier)

set(CMAKE_CXX_STANDARD 11)

# Task1
set(SOURCE_FILES
    src/simple_ml_ext.cpp
    src/simple_ml_ext.hpp
    src/softmax_classifier.cpp
)

# Create the executable 
add_executable(softmax ${SOURCE_FILES})

