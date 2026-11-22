cmake_minimum_required(VERSION 3.16)
project(tictactoe_vision CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui videoio dnn aruco calib3d)

add_library(ttt_core src/game.cpp src/minimax.cpp)
target_include_directories(ttt_core PUBLIC include)
target_compile_options(ttt_core PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>)

add_library(ttt_vision src/vision.cpp src/tracker.cpp)
target_link_libraries(ttt_vision PUBLIC ttt_core ${OpenCV_LIBS})
target_include_directories(ttt_vision PUBLIC ${OpenCV_INCLUDE_DIRS})

add_executable(ttt_vision_app src/main.cpp)
set_target_properties(ttt_vision_app PROPERTIES OUTPUT_NAME ttt_vision)
target_link_libraries(ttt_vision_app PRIVATE ttt_vision)

add_executable(ttt_bench src/bench.cpp)
target_link_libraries(ttt_bench PRIVATE ttt_vision)

option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(doctest GIT_REPOSITORY https://github.com/doctest/doctest.git GIT_TAG v2.4.11)
  FetchContent_MakeAvailable(doctest)
  enable_testing()
  add_executable(ttt_tests tests/test_game.cpp tests/test_tracker.cpp)
  target_link_libraries(ttt_tests PRIVATE ttt_vision doctest::doctest_with_main)
  add_test(NAME ttt_tests COMMAND ttt_tests)
endif()
