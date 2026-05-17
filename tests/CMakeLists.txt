cmake_minimum_required(VERSION 3.18)
project(ofxGgmlDiffusion_tests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(ofxGgmlDiffusion_tests
	test_main.cpp
	../../ofxGgmlCore/src/model/ofxGgmlModel.cpp
	../src/ofxGgmlDiffusion/ofxGgmlDiffusionAsyncRunner.cpp
	../src/ofxGgmlDiffusion/ofxGgmlDiffusionImageGenerationBackend.cpp
	../src/ofxGgmlDiffusion/ofxGgmlDiffusionGgufGanBackend.cpp
	../src/ofxGgmlDiffusion/ofxGgmlDiffusionTinyGanBackend.cpp
	../src/ofxGgmlDiffusion/ofxGgmlDiffusionTinyGanTraining.cpp
	../src/ofxGgmlDiffusion/ofxGgmlDiffusionNativeBackend.cpp
	../src/ofxGgmlDiffusion/ofxGgmlDiffusionUtils.cpp)

target_include_directories(ofxGgmlDiffusion_tests PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/../src
	${CMAKE_CURRENT_SOURCE_DIR}/../../ofxGgmlCore/src)

option(OFXGGMLDIFFUSION_BUILD_NATIVE_SMOKE "Build the optional stable-diffusion.cpp native bridge smoke test." OFF)
set(OFXGGMLDIFFUSION_STABLE_DIFFUSION_INCLUDE_DIR "" CACHE PATH "Path to generated stable-diffusion.cpp headers.")
set(OFXGGMLDIFFUSION_STABLE_DIFFUSION_LIBRARY "" CACHE FILEPATH "Path to generated stable-diffusion.cpp library.")
set(OFXGGMLDIFFUSION_GGML_INCLUDE_DIR "" CACHE PATH "Path to generated ggml headers from ofxGgmlCore.")
set(OFXGGMLDIFFUSION_EXTRA_LIBRARIES "" CACHE STRING "Extra libraries required by stable-diffusion.cpp.")

if (OFXGGMLDIFFUSION_BUILD_NATIVE_SMOKE)
	if (NOT OFXGGMLDIFFUSION_STABLE_DIFFUSION_INCLUDE_DIR)
		message(FATAL_ERROR "OFXGGMLDIFFUSION_STABLE_DIFFUSION_INCLUDE_DIR is required for the native smoke test.")
	endif()
	if (NOT OFXGGMLDIFFUSION_STABLE_DIFFUSION_LIBRARY)
		message(FATAL_ERROR "OFXGGMLDIFFUSION_STABLE_DIFFUSION_LIBRARY is required for the native smoke test.")
	endif()
	if (NOT OFXGGMLDIFFUSION_GGML_INCLUDE_DIR)
		message(FATAL_ERROR "OFXGGMLDIFFUSION_GGML_INCLUDE_DIR is required for the native smoke test.")
	endif()

	add_executable(ofxGgmlDiffusion_native_smoke
		test_native_smoke.cpp
		../src/ofxGgmlDiffusion/ofxGgmlDiffusionNativeBackend.cpp
		../src/ofxGgmlDiffusion/ofxGgmlDiffusionUtils.cpp)

	target_include_directories(ofxGgmlDiffusion_native_smoke PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/../src
		${OFXGGMLDIFFUSION_STABLE_DIFFUSION_INCLUDE_DIR}
		${OFXGGMLDIFFUSION_GGML_INCLUDE_DIR})
	target_compile_definitions(ofxGgmlDiffusion_native_smoke PRIVATE OFXGGMLDIFFUSION_WITH_STABLE_DIFFUSION)
	target_link_libraries(ofxGgmlDiffusion_native_smoke PRIVATE
		${OFXGGMLDIFFUSION_STABLE_DIFFUSION_LIBRARY}
		${OFXGGMLDIFFUSION_EXTRA_LIBRARIES})
	add_test(NAME ofxGgmlDiffusion_native_smoke COMMAND ofxGgmlDiffusion_native_smoke)
endif()

enable_testing()
add_test(NAME ofxGgmlDiffusion_tests COMMAND ofxGgmlDiffusion_tests)
