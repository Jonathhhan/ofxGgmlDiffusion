cmake_minimum_required(VERSION 3.18)
project(ofxGgmlDiffusion_tests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(ofxGgmlDiffusion_tests
	test_main.cpp
	../src/ofxGgmlDiffusion/ofxGgmlDiffusionNativeBackend.cpp
	../src/ofxGgmlDiffusion/ofxGgmlDiffusionUtils.cpp)

target_include_directories(ofxGgmlDiffusion_tests PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/../src)

enable_testing()
add_test(NAME ofxGgmlDiffusion_tests COMMAND ofxGgmlDiffusion_tests)
