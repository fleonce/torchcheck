cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_PREFIX_PATH "/home/moritz_hennen/libtorch/share/cmake/Torch")
set(CMAKE_MODULE_PATH "/home/moritz_hennen/Dokumente/NAWI/cuda_kernel/venv/lib/python3.10/site-packages/skbuild/resources/cmake")
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(LLTM LANGUAGES C CXX CUDA VERSION 12.1)
# if CUDA is optional: enable_language(CUDA)

find_package(PythonExtensions REQUIRED)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})

add_library(
        lltm_ext MODULE
        torch_check.cpp
)
python_extension_module(lltm_ext)
target_link_libraries(lltm_ext ${TORCH_LIBRARIES})
target_include_directories(lltm_ext PRIVATE ${TORCH_INCLUDE_DIRS})
set_property(TARGET lltm_ext PROPERTY CXX_STANDARD 17)

install(TARGETS lltm_ext DESTINATION lltm)