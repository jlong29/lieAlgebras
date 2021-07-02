cmake_minimum_required(VERSION 3.1)
project(eigen_mvn CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# LOCAL LIBRARIES
add_library(lieAlgebras
		lieAlgebras.h
		lieAlgebras.cpp
)

target_include_directories(
	lieAlgebras
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}
		$ENV{EIGEN_ROOT_DIR}
)

# Tests
#SO3
add_executable(SO3-SE3-Unit-Tests SO3-SE3-Unit-Tests.cpp)
set_target_properties(SO3-SE3-Unit-Tests PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(SO3-SE3-Unit-Tests PRIVATE -O3 -Wall -Wextra)
target_include_directories(
	SO3-SE3-Unit-Tests
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}
		$ENV{EIGEN_ROOT_DIR}
)
target_link_libraries(SO3-SE3-Unit-Tests lieAlgebras )

#SL3
add_executable(SL3-Unit-Tests SL3-Unit-Tests.cpp)
set_target_properties(SL3-Unit-Tests PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(SL3-Unit-Tests PRIVATE -O3 -Wall -Wextra)
target_include_directories(
	SL3-Unit-Tests
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}
		$ENV{EIGEN_ROOT_DIR}
)
target_link_libraries(SL3-Unit-Tests lieAlgebras )

#Adjoints
add_executable(Adjoint-Tests Adjoint-Tests.cpp)
set_target_properties(Adjoint-Tests PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(Adjoint-Tests PRIVATE -O3 -Wall -Wextra)
target_include_directories(
	Adjoint-Tests
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}
		$ENV{EIGEN_ROOT_DIR}
)
target_link_libraries(Adjoint-Tests lieAlgebras )

# Coordindate Transform
add_executable(CoordinateFrame-Sim CoordinateFrame-Sim.cpp)
set_target_properties(CoordinateFrame-Sim PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(CoordinateFrame-Sim PRIVATE -O3 -Wall -Wextra)
target_include_directories(
	CoordinateFrame-Sim
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}
		$ENV{EIGEN_ROOT_DIR}
)
target_link_libraries(CoordinateFrame-Sim lieAlgebras )

add_executable(CoordinateFrame-Sim2 CoordinateFrame-Sim2.cpp)
set_target_properties(CoordinateFrame-Sim2 PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(CoordinateFrame-Sim2 PRIVATE -O3 -Wall -Wextra)
target_include_directories(
	CoordinateFrame-Sim2
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}
		$ENV{EIGEN_ROOT_DIR}
)
target_link_libraries(CoordinateFrame-Sim2 lieAlgebras )

