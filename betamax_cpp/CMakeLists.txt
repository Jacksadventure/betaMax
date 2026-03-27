cmake_minimum_required(VERSION 3.16)

project(betamax_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(betamax_cpp
  src/main.cpp
)

target_compile_features(betamax_cpp PRIVATE cxx_std_17)

if (MSVC)
  target_compile_options(betamax_cpp PRIVATE /W4)
else()
  target_compile_options(betamax_cpp PRIVATE -Wall -Wextra -Wpedantic)
endif()

