cmake_minimum_required(VERSION 3.10)
Project(ISING)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(${PROJECT_NAME} ising.cxx)

target_compile_options(${PROJECT_NAME}
  PUBLIC -O3 -march=native -W -Wall -Wextra -pedantic -pedantic-errors)

get_target_property(MAIN_CXXFLAGS ${PROJECT_NAME} COMPILE_OPTIONS)

# also see: COMPILE_DEFINITIONS INCLUDE_DIRECTORIES
message("-- Target compiler flags are: ${MAIN_CXXFLAGS}")

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
COMMAND echo built with the flags: ${MAIN_CXXFLAGS})
