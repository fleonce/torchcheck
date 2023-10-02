cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/venv/lib/python3.10/site-packages/torch/share/cmake/Torch")
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/venv/lib/python3.10/site-packages/skbuild/resources/cmake")
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(TorchCheck LANGUAGES C CXX CUDA VERSION 12.1)
# if CUDA is optional: enable_language(CUDA)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/venv")
    find_package(Python3 3.10 EXACT COMPONENTS Interpreter)
    set(TORCH_CHECK_PYTHON_VERSION ${Python3_VERSION})
    find_program(VIRTUALENV virtualenv)
    execute_process(COMMAND "${VIRTUALENV}" "venv" "-p" "${Python3_VERSION}")
    SET(ENV{VIRTUAL_ENV} "${CMAKE_CURRENT_SOURCE_DIR}/venv")
    set(Python3_FIND_VIRTUALENV FIRST)
    unset (Python3_EXECUTABLE)
    find_package(Python3 3.10 EXACT COMPONENTS Interpreter)
    execute_process(COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/venv/bin/pip" "install" "-r" "requirements.txt")
endif()

find_package(PythonExtensions REQUIRED)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})

add_library(
        torchcheck_ext MODULE
        torchcheck/asserts.cpp
        torchcheck/index.cpp
        torchcheck/index.h
        torchcheck/main.cpp
        torchcheck/asserts.h
        torchcheck/pybind/torch_dtype.h
)
python_extension_module(torchcheck_ext)
target_link_libraries(torchcheck_ext ${TORCH_LIBRARIES})
target_include_directories(torchcheck_ext PRIVATE ${TORCH_INCLUDE_DIRS})
set_property(TARGET torchcheck_ext PROPERTY CXX_STANDARD 17)

install(TARGETS torchcheck_ext DESTINATION torchcheck)