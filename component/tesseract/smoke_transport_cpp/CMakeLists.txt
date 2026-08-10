cmake_minimum_required(VERSION 3.22)
project(smoke_transport_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# -ffast-math is deliberately absent: the adjoint dot-product test of
# specification section 6 requires IEEE-compliant double arithmetic.
add_compile_options(-O3 -Wall -Wextra)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# pybind11 ships its CMake package inside the wheel; ask the interpreter that
# will import the module where to find it.
if(NOT pybind11_DIR)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_LOOKUP_STATUS
  )
  if(NOT PYBIND11_LOOKUP_STATUS EQUAL 0)
    message(FATAL_ERROR "pybind11 is required to build smoke_transport_cpp")
  endif()
  set(pybind11_DIR "${PYBIND11_CMAKE_DIR}")
endif()
find_package(pybind11 CONFIG REQUIRED)

find_package(OpenMP REQUIRED)

pybind11_add_module(smoke_kernel_cpp
  cpp/smoke_kernel.cpp
  cpp/bindings.cpp
)
target_include_directories(smoke_kernel_cpp PRIVATE cpp)
target_link_libraries(smoke_kernel_cpp PRIVATE OpenMP::OpenMP_CXX)

install(TARGETS smoke_kernel_cpp DESTINATION .)
